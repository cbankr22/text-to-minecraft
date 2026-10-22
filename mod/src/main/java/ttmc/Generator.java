package ttmc;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Prompt -> 32^3 block tokens with the exported ONNX models (scripts/export_onnx.py). Two model
 * kinds, named in ttmc.json:
 *   "flow"     flow matching (ttmc/flow.py sample): Euler steps from noise to a latent, with
 *              classifier-free guidance; decoder.onnx rounds and decodes the latent
 *   "maskgit"  MaskGIT (OnnxBuilder.generate in ttmc/onnx_infer.py); decoder.onnx takes codes
 */
public final class Generator implements AutoCloseable {
    public final String[] blockNames;          // index = token, 0 = air
    public final int res;

    private final OrtEnvironment env = OrtEnvironment.getEnvironment();
    private final OrtSession clip, net, decoder;
    private final ClipTokenizer tokenizer;
    private final boolean flow;
    private final int seq, grid, context;
    private final int codebook, maskId;          // maskgit
    private final int latentDim, flowSteps;      // flow
    private final float flowCfg, flowShift;
    private Text empty;                         // CLIP features of "" (the unguided branch)

    public Generator(Path folder, int threads) throws IOException, OrtException {
        JsonObject meta = JsonParser.parseString(Files.readString(folder.resolve("ttmc.json"))).getAsJsonObject();
        flow = meta.has("kind") && meta.get("kind").getAsString().equals("flow");
        seq = meta.get("seq").getAsInt();
        codebook = flow ? 0 : meta.get("codebook").getAsInt();
        maskId = flow ? 0 : meta.get("mask_id").getAsInt();
        latentDim = flow ? meta.get("latent_dim").getAsInt() : 0;
        flowSteps = flow ? meta.get("steps").getAsInt() : 0;
        flowCfg = flow ? meta.get("cfg").getAsFloat() : 0;
        flowShift = flow ? meta.get("shift").getAsFloat() : 0;
        grid = meta.get("grid").getAsInt();
        res = meta.get("res").getAsInt();
        context = meta.get("context").getAsInt();
        JsonArray names = meta.getAsJsonArray("blocks");
        blockNames = new String[names.size()];
        for (int i = 0; i < blockNames.length; i++) blockNames[i] = names.get(i).getAsString();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setIntraOpNumThreads(threads);
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        clip = env.createSession(folder.resolve("clip_text.onnx").toString(), opts);
        net = env.createSession(folder.resolve(flow ? "flow.onnx" : "t2v.onnx").toString(), opts);
        decoder = env.createSession(folder.resolve("decoder.onnx").toString(), opts);
        tokenizer = new ClipTokenizer(folder, context);
    }

    /** CLIP hidden states for one prompt, flattened (context * dim), plus its padding mask. */
    private record Text(float[] hidden, boolean[] pad) {}

    private Text encodeText(String prompt) throws OrtException {
        long[][] enc = tokenizer.encode(prompt);
        long[] shape = {1, context};
        try (OnnxTensor ids = OnnxTensor.createTensor(env, LongBuffer.wrap(enc[0]), shape);
             OnnxTensor mask = OnnxTensor.createTensor(env, LongBuffer.wrap(enc[1]), shape);
             OrtSession.Result r = clip.run(Map.of("input_ids", ids, "attention_mask", mask))) {
            FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
            float[] hidden = new float[fb.remaining()];
            fb.get(hidden);
            boolean[] pad = new boolean[context];
            for (int i = 0; i < context; i++) pad[i] = enc[1][i] == 0;
            return new Text(hidden, pad);
        }
    }

    /** Block tokens in (Y, Z, X) order, length res^3, with each model kind's default settings
     *  (flow: from ttmc.json; MaskGIT: the best of the 2026-10-08 sampling grid). */
    public byte[] generate(String prompt, long seed) throws OrtException {
        return flow ? generateFlow(prompt, flowSteps, flowCfg, flowShift, seed)
                    : generate(prompt, 32, 7.0f, 0.8f, 2.0f, seed);
    }

    /** CLIP features of the prompt (row 0) and of "" (row 1), as one batch of 2. */
    private OnnxTensor[] textBatch(String prompt) throws OrtException {
        if (empty == null) empty = encodeText("");
        Text cond = encodeText(prompt), uncond = empty;
        int dim = cond.hidden().length / context;
        float[] textH = new float[2 * context * dim];
        System.arraycopy(cond.hidden(), 0, textH, 0, context * dim);
        System.arraycopy(uncond.hidden(), 0, textH, context * dim, context * dim);
        boolean[][] textPad = {cond.pad(), uncond.pad()};
        return new OnnxTensor[]{OnnxTensor.createTensor(env, FloatBuffer.wrap(textH), new long[]{2, context, dim}),
                OnnxTensor.createTensor(env, textPad)};
    }

    /** Flow matching: Euler integration from noise (t = 1) to a latent (t = 0), same as ttmc/flow.py sample. */
    public byte[] generateFlow(String prompt, int steps, float cfg, float shift, long seed) throws OrtException {
        SplittableRandom rng = new SplittableRandom(seed);
        int n = seq * latentDim;
        float[] z = new float[n];
        for (int i = 0; i < n; i++) z[i] = (float) rng.nextGaussian();
        float[] ts = new float[steps + 1];
        for (int i = 0; i <= steps; i++) {                 // evenly spaced, then shifted toward high noise
            double s = 1 - i / (double) steps;
            ts[i] = (float) (shift * s / (1 + (shift - 1) * s));
        }
        OnnxTensor[] text = textBatch(prompt);
        try (OnnxTensor hT = text[0]; OnnxTensor padT = text[1]) {
            float[] both = new float[2 * n];
            for (int i = 0; i < steps; i++) {
                System.arraycopy(z, 0, both, 0, n);
                System.arraycopy(z, 0, both, n, n);
                try (OnnxTensor zT = OnnxTensor.createTensor(env, FloatBuffer.wrap(both), new long[]{2, seq, latentDim});
                     OnnxTensor tT = OnnxTensor.createTensor(env, FloatBuffer.wrap(new float[]{ts[i], ts[i]}), new long[]{2});
                     OrtSession.Result r = net.run(Map.of("latent", zT, "t", tT, "text_h", hT, "text_pad", padT))) {
                    FloatBuffer v = ((OnnxTensor) r.get(0)).getFloatBuffer();
                    float dt = ts[i + 1] - ts[i];
                    for (int k = 0; k < n; k++) {
                        float vc = v.get(k), vu = v.get(n + k);    // row 0 guided by the prompt, row 1 unguided
                        z[k] += dt * (vu + cfg * (vc - vu));
                    }
                }
            }
        }
        try (OnnxTensor latent = OnnxTensor.createTensor(env, FloatBuffer.wrap(z), new long[]{1, seq, latentDim});
             OrtSession.Result r = decoder.run(Map.of("latent", latent))) {
            return blocks(r);
        }
    }

    private byte[] blocks(OrtSession.Result r) {
        ByteBuffer bb = ((OnnxTensor) r.get(0)).getByteBuffer();
        byte[] blocks = new byte[res * res * res];
        bb.get(blocks);
        return blocks;
    }

    /** MaskGIT with classifier-free guidance. */
    public byte[] generate(String prompt, int steps, float cfg, float temperature, float choiceTemp, long seed)
            throws OrtException {
        SplittableRandom rng = new SplittableRandom(seed);

        long[] tokens = new long[seq];
        Arrays.fill(tokens, maskId);
        double[] probs = new double[codebook];
        long[] sampled = new long[seq];
        double[] conf = new double[seq];

        OnnxTensor[] text = textBatch(prompt);
        try (OnnxTensor hT = text[0]; OnnxTensor padT = text[1]) {
            for (int t = 0; t < steps; t++) {
                long[] both = new long[2 * seq];
                System.arraycopy(tokens, 0, both, 0, seq);
                System.arraycopy(tokens, 0, both, seq, seq);
                float[] logits;
                try (OnnxTensor tokT = OnnxTensor.createTensor(env, LongBuffer.wrap(both), new long[]{2, seq});
                     OrtSession.Result r = net.run(Map.of("tokens", tokT, "text_h", hT, "text_pad", padT))) {
                    FloatBuffer fb = ((OnnxTensor) r.get(0)).getFloatBuffer();
                    logits = new float[fb.remaining()];
                    fb.get(logits);
                }
                int off = seq * codebook;                     // start of the unguided row
                for (int p = 0; p < seq; p++) {
                    // guided logits -> softmax -> sample one code by inverse CDF
                    int base = p * codebook;
                    double max = Double.NEGATIVE_INFINITY;
                    for (int k = 0; k < codebook; k++) {
                        float u = logits[off + base + k];
                        double g = (u + cfg * (logits[base + k] - u)) / temperature;
                        probs[k] = g;
                        if (g > max) max = g;
                    }
                    double sum = 0;
                    for (int k = 0; k < codebook; k++) sum += probs[k] = Math.exp(probs[k] - max);
                    double target = rng.nextDouble() * sum, acc = 0;
                    int pick = codebook - 1;
                    for (int k = 0; k < codebook; k++) {
                        acc += probs[k];
                        if (acc >= target) { pick = k; break; }
                    }
                    boolean unknown = tokens[p] == maskId;
                    sampled[p] = unknown ? pick : tokens[p];
                    conf[p] = probs[pick] / sum;
                }
                int nMask = (int) Math.floor(Math.cos(Math.PI / 2 * (t + 1) / steps) * seq);
                if (nMask == 0) {
                    tokens = sampled.clone();
                    break;
                }
                // keep the most confident codes; re-mask the rest (with annealed randomness)
                double anneal = choiceTemp * (1 - (t + 1) / (double) steps);
                Integer[] order = new Integer[seq];
                double[] score = new double[seq];
                for (int p = 0; p < seq; p++) {
                    order[p] = p;
                    if (tokens[p] != maskId) {
                        score[p] = Double.POSITIVE_INFINITY;
                    } else {
                        double gumbel = -Math.log(-Math.log(Math.max(rng.nextDouble(), 1e-20)));
                        score[p] = Math.log(Math.max(conf[p], 1e-20)) + anneal * gumbel;
                    }
                }
                Arrays.sort(order, (a, b) -> Double.compare(score[a], score[b]));
                tokens = sampled.clone();
                for (int i = 0; i < nMask; i++) tokens[order[i]] = maskId;
            }
        }

        try (OnnxTensor codes = OnnxTensor.createTensor(env, LongBuffer.wrap(tokens), new long[]{1, grid, grid, grid});
             OrtSession.Result r = decoder.run(Map.of("codes", codes))) {
            return blocks(r);
        }
    }

    @Override
    public void close() throws OrtException {
        clip.close();
        net.close();
        decoder.close();
    }
}
