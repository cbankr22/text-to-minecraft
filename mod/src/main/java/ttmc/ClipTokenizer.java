package ttmc;

import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * CLIP's byte-level BPE tokenizer. A port of ClipTokenizer in ttmc/onnx_infer.py, which
 * matches Hugging Face's CLIPTokenizer on 5,006 of 5,006 test captions.
 */
public final class ClipTokenizer {
    private static final Pattern WORDS = Pattern.compile(
            "<\\|startoftext\\|>|<\\|endoftext\\|>|'s|'t|'re|'ve|'m|'ll|'d|[\\p{L}]+|[\\p{N}]|[^\\s\\p{L}\\p{N}]+",
            Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE);

    private final Map<String, Integer> encoder = new HashMap<>();
    private final Map<String, Integer> ranks = new HashMap<>();   // "a b" -> merge rank
    private final Map<String, List<String>> cache = new HashMap<>();
    private final String[] byteToChar = new String[256];
    private final int bos, eos, context;

    public ClipTokenizer(Path folder, int context) throws IOException {
        JsonObject vocab = JsonParser.parseString(Files.readString(folder.resolve("vocab.json"))).getAsJsonObject();
        for (var e : vocab.entrySet()) encoder.put(e.getKey(), e.getValue().getAsInt());
        List<String> merges = Files.readAllLines(folder.resolve("merges.txt"), StandardCharsets.UTF_8);
        for (int i = 1; i < merges.size(); i++) {               // line 0 is a version header
            if (!merges.get(i).isEmpty()) ranks.put(merges.get(i), i - 1);
        }
        bos = encoder.get("<|startoftext|>");
        eos = encoder.get("<|endoftext|>");
        this.context = context;
        buildByteTable();
    }

    /** GPT-2/CLIP byte <-> printable character table. */
    private void buildByteTable() {
        boolean[] printable = new boolean[256];
        for (int b = '!'; b <= '~'; b++) printable[b] = true;
        for (int b = 0xA1; b <= 0xAC; b++) printable[b] = true;
        for (int b = 0xAE; b <= 0xFF; b++) printable[b] = true;
        int n = 0;
        for (int b = 0; b < 256; b++) {
            byteToChar[b] = printable[b] ? String.valueOf((char) b) : String.valueOf((char) (256 + n++));
        }
    }

    private List<String> bpe(String token) {
        List<String> cached = cache.get(token);
        if (cached != null) return cached;
        List<String> word = new ArrayList<>();
        int[] cps = token.codePoints().toArray();
        for (int i = 0; i < cps.length; i++) {
            String c = new String(cps, i, 1);
            word.add(i == cps.length - 1 ? c + "</w>" : c);
        }
        while (word.size() > 1) {
            int best = Integer.MAX_VALUE;
            String a = null, b = null;
            for (int i = 0; i < word.size() - 1; i++) {
                Integer r = ranks.get(word.get(i) + " " + word.get(i + 1));
                if (r != null && r < best) {
                    best = r;
                    a = word.get(i);
                    b = word.get(i + 1);
                }
            }
            if (a == null) break;
            List<String> merged = new ArrayList<>();
            for (int i = 0; i < word.size(); ) {               // merge every (a, b), left to right
                if (i < word.size() - 1 && word.get(i).equals(a) && word.get(i + 1).equals(b)) {
                    merged.add(a + b);
                    i += 2;
                } else {
                    merged.add(word.get(i++));
                }
            }
            word = merged;
        }
        cache.put(token, word);
        return word;
    }

    /** Returns {input_ids, attention_mask}, each of length context, padded with eos like HF. */
    public long[][] encode(String text) {
        text = unescapeHtml(unescapeHtml(text)).replaceAll("\\s+", " ").strip().toLowerCase(Locale.ROOT);
        List<Long> ids = new ArrayList<>();
        Matcher m = WORDS.matcher(text);
        while (m.find()) {
            StringBuilder sb = new StringBuilder();
            for (byte by : m.group().getBytes(StandardCharsets.UTF_8)) sb.append(byteToChar[by & 0xFF]);
            for (String piece : bpe(sb.toString())) ids.add((long) encoder.get(piece));
        }
        if (ids.size() > context - 2) ids = ids.subList(0, context - 2);
        long[] inputIds = new long[context];
        long[] mask = new long[context];
        Arrays.fill(inputIds, eos);
        inputIds[0] = bos;
        for (int i = 0; i < ids.size(); i++) inputIds[i + 1] = ids.get(i);
        Arrays.fill(mask, 0, ids.size() + 2, 1);
        return new long[][]{inputIds, mask};
    }

    /** The few entities html.unescape would change in a typed prompt. */
    private static String unescapeHtml(String s) {
        if (s.indexOf('&') < 0) return s;
        return s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"")
                .replace("&#39;", "'").replace("&apos;", "'").replace("&nbsp;", " ").replace("&amp;", "&");
    }
}
