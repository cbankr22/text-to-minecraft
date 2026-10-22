package ttmc;

import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Build size control, a port of ttmc/resize.py. The model always fills its 32^3 canvas;
 * the result is cropped and scaled so it measures the size asked for in the prompt.
 */
public final class Sizing {
    public static final int DEFAULT = 32, MAX = 64;
    private static final Map<String, Integer> WORDS = Map.of(
            "tiny", 8, "little", 12, "small", 16, "medium", 24, "big", 32, "large", 32,
            "huge", 48, "giant", 64, "gigantic", 64, "massive", 64);
    private static final Pattern NUM = Pattern.compile(
            ",?\\s*\\b(\\d{1,3})\\s*(?:blocks?|b)\\b\\s*(tall|high|wide|long|big|across)?",
            Pattern.CASE_INSENSITIVE);

    /** axis: "longest", "height" or "width" (longest horizontal side). */
    public record Size(int blocks, String axis) {}

    public record Parsed(String prompt, Size size) {}

    public static Parsed parse(String prompt) {
        Matcher m = NUM.matcher(prompt);
        if (m.find()) {
            String word = m.group(2) == null ? "" : m.group(2).toLowerCase(Locale.ROOT);
            String axis = switch (word) {
                case "tall", "high" -> "height";
                case "wide", "long", "across" -> "width";
                default -> "longest";
            };
            String clean = (prompt.substring(0, m.start()) + prompt.substring(m.end())).strip();
            clean = clean.replaceAll("^[ ,]+|[ ,]+$", "");
            return new Parsed(clean, new Size(Math.max(1, Math.min(Integer.parseInt(m.group(1)), MAX)), axis));
        }
        for (String w : prompt.toLowerCase(Locale.ROOT).split("[^a-z]+")) {
            Integer s = WORDS.get(w);
            if (s != null) return new Parsed(prompt, new Size(s, "longest"));
        }
        return new Parsed(prompt, new Size(DEFAULT, "longest"));
    }

    /** A cropped, resized build: tokens in (Y, Z, X) order with the given dimensions. */
    public record Build(byte[] blocks, int sy, int sz, int sx) {
        public int at(int y, int z, int x) { return blocks[(y * sz + z) * sx + x] & 0xFF; }
    }

    /**
     * Crop to the occupied box and scale uniformly so size.axis measures size.blocks.
     * Shrinking: a cell is solid if at least minFill of the voxels it covers are, and takes
     * their most common block. Growing: nearest neighbor.
     */
    public static Build resize(byte[] grid, int res, Size size, double minFill, int nBlocks) {
        int y0 = res, y1 = -1, z0 = res, z1 = -1, x0 = res, x1 = -1;
        for (int y = 0; y < res; y++)
            for (int z = 0; z < res; z++)
                for (int x = 0; x < res; x++)
                    if (grid[(y * res + z) * res + x] != 0) {
                        y0 = Math.min(y0, y); y1 = Math.max(y1, y);
                        z0 = Math.min(z0, z); z1 = Math.max(z1, z);
                        x0 = Math.min(x0, x); x1 = Math.max(x1, x);
                    }
        if (y1 < 0) return new Build(new byte[0], 0, 0, 0);
        int Y = y1 - y0 + 1, Z = z1 - z0 + 1, X = x1 - x0 + 1;
        int ref = switch (size.axis()) {
            case "height" -> Y;
            case "width" -> Math.max(Z, X);
            default -> Math.max(Y, Math.max(Z, X));
        };
        double s = Math.min(size.blocks() / (double) ref, MAX / (double) Math.max(Y, Math.max(Z, X)));
        int ny = Math.max(1, (int) Math.round(Y * s)), nz = Math.max(1, (int) Math.round(Z * s)),
                nx = Math.max(1, (int) Math.round(X * s));
        byte[] out = new byte[ny * nz * nx];
        if (s >= 1) {
            for (int y = 0; y < ny; y++)
                for (int z = 0; z < nz; z++)
                    for (int x = 0; x < nx; x++) {
                        int sy = Math.min((int) (y / s), Y - 1), sz = Math.min((int) (z / s), Z - 1),
                                sx = Math.min((int) (x / s), X - 1);
                        out[(y * nz + z) * nx + x] = grid[((y0 + sy) * res + z0 + sz) * res + x0 + sx];
                    }
            return new Build(out, ny, nz, nx);
        }
        int cells = ny * nz * nx;
        int[] volume = new int[cells];
        int[] counts = new int[cells * nBlocks];
        for (int y = 0; y < Y; y++)
            for (int z = 0; z < Z; z++)
                for (int x = 0; x < X; x++) {
                    int c = (Math.min((int) (y * s), ny - 1) * nz + Math.min((int) (z * s), nz - 1)) * nx
                            + Math.min((int) (x * s), nx - 1);
                    volume[c]++;
                    int b = grid[((y0 + y) * res + z0 + z) * res + x0 + x] & 0xFF;
                    if (b != 0) counts[c * nBlocks + b]++;          // air never wins the vote
                }
        for (int c = 0; c < cells; c++) {
            int solid = 0, best = 0, bestN = 0;
            for (int b = 1; b < nBlocks; b++) {
                int n = counts[c * nBlocks + b];
                solid += n;
                if (n > bestN) { bestN = n; best = b; }
            }
            if (solid >= minFill * volume[c]) out[c] = (byte) best;
        }
        return new Build(out, ny, nz, nx);
    }
}
