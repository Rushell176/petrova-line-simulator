# Petrova Line Simulator

A native Rust/wgpu recreation of the Petrova Line scene from *Project Hail Mary*, presented as a walkable 360-degree simulation.

## Play on Windows

Download [Petrova-Line-Simulator-v0.1.0-windows-x64.zip](release/Petrova-Line-Simulator-v0.1.0-windows-x64.zip), extract it, and run `Petrova Line Simulator.exe`.

Requirements:

- Windows 10 or 11, 64-bit
- A DirectX 12, Vulkan, or OpenGL-capable GPU with current drivers

The projector starts off. Press `P`, or aim at the projector and click it, to fade in the red light field.

### Controls

| Input | Action |
| --- | --- |
| Mouse | Look around |
| `W` `A` `S` `D` | Move |
| `Space` | Move up |
| `Shift` | Move down |
| `P` | Toggle the projected lights |
| Click projector | Toggle the projected lights while aiming at it |
| `F11` | Toggle fullscreen |
| `Esc` | Release the mouse; press again to close |

See [How to Play.txt](release/How%20to%20Play.txt) for the player-facing instructions.

## Build from source

Install the Rust toolchain, clone this repository, and run:

```sh
cargo run --release
```

For an optimized Windows build:

```powershell
cargo build --release --target x86_64-pc-windows-msvc
```

The executable is written to `target/x86_64-pc-windows-msvc/release/petrove_line.exe`.

To build a Windows executable with the C runtime statically linked:

```powershell
$env:RUSTFLAGS = '-C target-feature=+crt-static'
cargo build --release --target x86_64-pc-windows-msvc
```

## Repository layout

- `src/` — Rust application, procedural geometry, and WGSL shaders
- `release/` — tested Windows executable, player instructions, and downloadable ZIP
- `Cargo.toml` / `Cargo.lock` — Rust package metadata and locked dependencies

## License

The original source code is available under the [MIT License](LICENSE).

This is an unofficial fan-made recreation inspired by *Project Hail Mary*. It is not affiliated with or endorsed by the author, filmmakers, publishers, or other rights holders. The MIT License applies to this repository's original code, not to the underlying novel, film, characters, names, or other third-party rights.
