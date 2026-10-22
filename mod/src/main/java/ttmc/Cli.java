package ttmc;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;

/**
 * Checks the Java pipeline outside Minecraft:
 *   java -cp <classes + onnxruntime + gson> ttmc.Cli <model dir> <out.bin> "prompt" [seed]
 * Prints the CLIP token ids (compare with ttmc/onnx_infer.py) and writes the resized build as
 * 3 ints (sy, sz, sx) followed by the block tokens in (Y, Z, X) order.
 */
public final class Cli {
    public static void main(String[] args) throws Exception {
        Path dir = Path.of(args[0]);
        String prompt = args[2];
        long seed = args.length > 3 ? Long.parseLong(args[3]) : 0;
        ClipTokenizer tok = new ClipTokenizer(dir, 77);
        long[] ids = tok.encode(Sizing.parse(prompt).prompt())[0];
        int n = 0;
        while (n < ids.length && !(ids[n] == 49407 && n > 0)) n++;
        System.out.println("ids " + Arrays.toString(Arrays.copyOf(ids, n + 1)));

        try (Generator g = new Generator(dir, Math.max(1, Runtime.getRuntime().availableProcessors() / 2))) {
            Sizing.Parsed parsed = Sizing.parse(prompt);
            long t = System.currentTimeMillis();
            byte[] grid = g.generate(parsed.prompt(), seed);
            Sizing.Build b = Sizing.resize(grid, g.res, parsed.size(), 0.1, g.blockNames.length);
            System.out.printf("generated in %.1f s, size %dx%dx%d (y,z,x %d %d %d)%n",
                    (System.currentTimeMillis() - t) / 1000.0, b.sx(), b.sy(), b.sz(), b.sy(), b.sz(), b.sx());
            byte[] out = new byte[12 + b.blocks().length];
            java.nio.ByteBuffer.wrap(out).putInt(b.sy()).putInt(b.sz()).putInt(b.sx()).put(b.blocks());
            Files.write(Path.of(args[1]), out);
        }
    }
}
