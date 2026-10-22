package ttmc;

import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.exceptions.CommandSyntaxException;
import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;
import net.fabricmc.loader.api.FabricLoader;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.commands.Commands;
import net.minecraft.core.BlockPos;
import net.minecraft.core.Direction;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.network.chat.Component;
import net.minecraft.resources.Identifier;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.block.state.BlockState;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadLocalRandom;

/**
 * /ttmc <prompt>   generate a build and place it in front of you
 * /ttmc undo       remove your last build (restores what was there)
 *
 * Models live in config/ttmc/ (the files from scripts/export_onnx.py). Generation runs on its
 * own thread, one build at a time; blocks are placed on the server thread.
 */
public class TtmcMod implements ModInitializer {
    public static final Logger LOG = LoggerFactory.getLogger("ttmc");
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "ttmc-generator");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    });
    private static final int UNDO_DEPTH = 10;
    private static final Map<UUID, Deque<List<Placed>>> UNDO = new ConcurrentHashMap<>();
    private static volatile Generator generator;
    private static BlockState[] palette;

    private record Placed(BlockPos pos, BlockState old) {}

    @Override
    public void onInitialize() {
        CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) ->
                dispatcher.register(Commands.literal("ttmc")
                        .requires(Commands.hasPermission(Commands.LEVEL_GAMEMASTERS))
                        .then(Commands.literal("undo").executes(ctx -> undo(ctx.getSource())))
                        .then(Commands.argument("prompt", StringArgumentType.greedyString())
                                .executes(ctx -> build(ctx.getSource(), StringArgumentType.getString(ctx, "prompt"))))));
    }

    static Path modelDir() {
        return FabricLoader.getInstance().getConfigDir().resolve("ttmc");
    }

    private static synchronized Generator generator() throws Exception {
        if (generator == null) {
            Path dir = modelDir();
            if (!Files.exists(dir.resolve("ttmc.json"))) {
                throw new IllegalStateException("model files not found in " + dir.toAbsolutePath());
            }
            long t = System.currentTimeMillis();
            int threads = Math.max(1, Runtime.getRuntime().availableProcessors() / 2);
            generator = new Generator(dir, threads);
            LOG.info("TTMC models loaded in {} ms ({} threads)", System.currentTimeMillis() - t, threads);
        }
        return generator;
    }

    private static int build(CommandSourceStack source, String text) throws CommandSyntaxException {
        ServerPlayer player = source.getPlayerOrException();
        MinecraftServer server = source.getServer();
        Sizing.Parsed parsed = Sizing.parse(text);
        // where to build is decided now, from where the player stands when they ask
        BlockPos feet = player.blockPosition();
        Direction facing = player.getDirection();
        ServerLevel level = (ServerLevel) player.level();
        UUID id = player.getUUID();
        source.sendSuccess(() -> Component.literal("Building \"" + parsed.prompt() + "\"..."), false);

        WORKER.submit(() -> {
            try {
                long t = System.currentTimeMillis();
                Generator g = generator();
                byte[] grid = g.generate(parsed.prompt(), ThreadLocalRandom.current().nextLong());
                Sizing.Build b = Sizing.resize(grid, g.res, parsed.size(), 0.1, g.blockNames.length);
                double secs = (System.currentTimeMillis() - t) / 1000.0;
                server.execute(() -> {
                    int n = place(level, feet, facing, b, g.blockNames, id);
                    source.sendSuccess(() -> Component.literal(String.format(
                            "Placed %d blocks (%dx%dx%d) in %.1f s. /ttmc undo removes it.", n, b.sx(), b.sy(), b.sz(), secs)), false);
                });
            } catch (Throwable e) {
                LOG.error("TTMC build failed", e);
                server.execute(() -> source.sendFailure(Component.literal("TTMC: " + e.getMessage())));
            }
        });
        return 1;
    }

    /**
     * Height to build on: the ground surface at the build's center, so it neither floats nor
     * sinks into a slope. Underground or indoors the surface is far from the player (the roof,
     * the sky), so then it stays at the player's feet.
     */
    private static int groundY(ServerLevel level, int x, int z, int feetY) {
        int surface = level.getHeight(Heightmap.Types.MOTION_BLOCKING_NO_LEAVES, x, z);
        return Math.abs(surface - feetY) <= 8 ? surface : feetY;
    }

    /** Places a build 2 blocks in front of the player, centered on their line of sight, on the ground. */
    private static int place(ServerLevel level, BlockPos feet, Direction facing, Sizing.Build b, String[] names, UUID id) {
        if (palette == null) {
            palette = new BlockState[names.length];
            for (int i = 0; i < names.length; i++) {
                palette[i] = BuiltInRegistries.BLOCK.getValue(Identifier.parse(names[i])).defaultBlockState();
            }
        }
        int dist = 2 + Math.max(b.sx(), b.sz()) / 2;
        int cx = feet.getX() + facing.getStepX() * dist, cz = feet.getZ() + facing.getStepZ() * dist;
        int x0 = cx - b.sx() / 2, z0 = cz - b.sz() / 2, y0 = groundY(level, cx, cz, feet.getY());
        List<Placed> placed = new ArrayList<>();
        for (int y = 0; y < b.sy(); y++)
            for (int z = 0; z < b.sz(); z++)
                for (int x = 0; x < b.sx(); x++) {
                    int token = b.at(y, z, x);
                    if (token == 0) continue;
                    BlockPos pos = new BlockPos(x0 + x, y0 + y, z0 + z);
                    placed.add(new Placed(pos, level.getBlockState(pos)));
                    level.setBlock(pos, palette[token], Block.UPDATE_CLIENTS);
                }
        Deque<List<Placed>> history = UNDO.computeIfAbsent(id, k -> new ArrayDeque<>());
        history.push(placed);
        while (history.size() > UNDO_DEPTH) history.removeLast();
        return placed.size();
    }

    private static int undo(CommandSourceStack source) throws CommandSyntaxException {
        ServerPlayer player = source.getPlayerOrException();
        Deque<List<Placed>> history = UNDO.get(player.getUUID());
        if (history == null || history.isEmpty()) {
            source.sendFailure(Component.literal("Nothing to undo."));
            return 0;
        }
        ServerLevel level = (ServerLevel) player.level();
        List<Placed> last = history.pop();
        for (int i = last.size() - 1; i >= 0; i--) {
            level.setBlock(last.get(i).pos(), last.get(i).old(), Block.UPDATE_CLIENTS);
        }
        source.sendSuccess(() -> Component.literal("Removed " + last.size() + " blocks."), false);
        return 1;
    }
}
