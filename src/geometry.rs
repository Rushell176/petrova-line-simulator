use glam::Vec3;

pub const TAU: f32 = std::f32::consts::TAU;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub center: [f32; 3],
    pub size: f32,
    pub seed: f32,
    pub speed: f32,
    pub tint: [f32; 3],
    pub _pad: f32,
}

pub struct MeshBuilder {
    pub verts: Vec<Vertex>,
    pub idx: Vec<u32>,
}

impl MeshBuilder {
    pub fn new() -> Self {
        Self {
            verts: Vec::new(),
            idx: Vec::new(),
        }
    }

    fn v(&mut self, pos: Vec3, normal: Vec3, color: [f32; 4]) -> u32 {
        let i = self.verts.len() as u32;
        self.verts.push(Vertex {
            pos: pos.to_array(),
            normal: normal.to_array(),
            color,
        });
        i
    }

    fn quad(&mut self, a: Vec3, b: Vec3, c: Vec3, d: Vec3, normal: Vec3, color: [f32; 4]) {
        let i = self.v(a, normal, color);
        self.v(b, normal, color);
        self.v(c, normal, color);
        self.v(d, normal, color);
        self.idx
            .extend_from_slice(&[i, i + 1, i + 2, i, i + 2, i + 3]);
    }

    pub fn cylinder(
        &mut self,
        p0: Vec3,
        p1: Vec3,
        r0: f32,
        r1: f32,
        seg: usize,
        color: [f32; 4],
        caps: bool,
    ) {
        let axis_v = p1 - p0;
        let len = axis_v.length();
        if len < 1e-5 {
            return;
        }
        let axis = axis_v / len;
        let helper = if axis.y.abs() < 0.95 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u = axis.cross(helper).normalize();
        let w = axis.cross(u).normalize();
        let side_normal = |radial: Vec3| (radial * len - axis * (r1 - r0)).normalize();

        for i in 0..seg {
            let a0 = TAU * i as f32 / seg as f32;
            let a1 = TAU * (i + 1) as f32 / seg as f32;
            let d0 = u * a0.cos() + w * a0.sin();
            let d1 = u * a1.cos() + w * a1.sin();
            let n0 = side_normal(d0);
            let n1 = side_normal(d1);
            let i0 = self.v(p0 + d0 * r0, n0, color);
            let i1 = self.v(p0 + d1 * r0, n1, color);
            let i2 = self.v(p1 + d1 * r1, n1, color);
            let i3 = self.v(p1 + d0 * r1, n0, color);
            self.idx.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
        }

        if caps {
            if r0 > 1e-4 {
                let c = self.v(p0, -axis, color);
                for i in 0..seg {
                    let a0 = TAU * i as f32 / seg as f32;
                    let a1 = TAU * (i + 1) as f32 / seg as f32;
                    let d0 = u * a0.cos() + w * a0.sin();
                    let d1 = u * a1.cos() + w * a1.sin();
                    let i0 = self.v(p0 + d0 * r0, -axis, color);
                    let i1 = self.v(p0 + d1 * r0, -axis, color);
                    self.idx.extend_from_slice(&[c, i0, i1]);
                }
            }
            if r1 > 1e-4 {
                let c = self.v(p1, axis, color);
                for i in 0..seg {
                    let a0 = TAU * i as f32 / seg as f32;
                    let a1 = TAU * (i + 1) as f32 / seg as f32;
                    let d0 = u * a0.cos() + w * a0.sin();
                    let d1 = u * a1.cos() + w * a1.sin();
                    let i0 = self.v(p1 + d0 * r1, axis, color);
                    let i1 = self.v(p1 + d1 * r1, axis, color);
                    self.idx.extend_from_slice(&[c, i0, i1]);
                }
            }
        }
    }

    pub fn torus(
        &mut self,
        center: Vec3,
        axis: Vec3,
        radius: f32,
        tube: f32,
        major: usize,
        minor: usize,
        color: [f32; 4],
    ) {
        let axis = axis.normalize();
        let helper = if axis.y.abs() < 0.95 {
            Vec3::Y
        } else {
            Vec3::X
        };
        let u = axis.cross(helper).normalize();
        let w = axis.cross(u).normalize();

        let point = |i: usize, j: usize| -> (Vec3, Vec3) {
            let th = TAU * i as f32 / major as f32;
            let ph = TAU * j as f32 / minor as f32;
            let ring = u * th.cos() + w * th.sin();
            let pos = center + ring * (radius + tube * ph.cos()) + axis * (tube * ph.sin());
            let n = (ring * ph.cos() + axis * ph.sin()).normalize();
            (pos, n)
        };

        for i in 0..major {
            for j in 0..minor {
                let (p00, n00) = point(i, j);
                let (p10, n10) = point(i + 1, j);
                let (p11, n11) = point(i + 1, j + 1);
                let (p01, n01) = point(i, j + 1);
                let i0 = self.v(p00, n00, color);
                let i1 = self.v(p10, n10, color);
                let i2 = self.v(p11, n11, color);
                let i3 = self.v(p01, n01, color);
                self.idx.extend_from_slice(&[i0, i1, i2, i0, i2, i3]);
            }
        }
    }

    pub fn cuboid(&mut self, center: Vec3, half: Vec3, color: [f32; 4]) {
        let v =
            |sx: f32, sy: f32, sz: f32| center + Vec3::new(half.x * sx, half.y * sy, half.z * sz);
        self.quad(
            v(1.0, -1.0, -1.0),
            v(1.0, 1.0, -1.0),
            v(1.0, 1.0, 1.0),
            v(1.0, -1.0, 1.0),
            Vec3::X,
            color,
        );
        self.quad(
            v(-1.0, -1.0, 1.0),
            v(-1.0, 1.0, 1.0),
            v(-1.0, 1.0, -1.0),
            v(-1.0, -1.0, -1.0),
            -Vec3::X,
            color,
        );
        self.quad(
            v(-1.0, 1.0, -1.0),
            v(-1.0, 1.0, 1.0),
            v(1.0, 1.0, 1.0),
            v(1.0, 1.0, -1.0),
            Vec3::Y,
            color,
        );
        self.quad(
            v(-1.0, -1.0, 1.0),
            v(-1.0, -1.0, -1.0),
            v(1.0, -1.0, -1.0),
            v(1.0, -1.0, 1.0),
            -Vec3::Y,
            color,
        );
        self.quad(
            v(-1.0, -1.0, 1.0),
            v(1.0, -1.0, 1.0),
            v(1.0, 1.0, 1.0),
            v(-1.0, 1.0, 1.0),
            Vec3::Z,
            color,
        );
        self.quad(
            v(-1.0, 1.0, -1.0),
            v(1.0, 1.0, -1.0),
            v(1.0, -1.0, -1.0),
            v(-1.0, -1.0, -1.0),
            -Vec3::Z,
            color,
        );
    }
}

pub const PROJECTOR_POS: [f32; 3] = [0.0, 4.92, 2.0];
pub const PROJECTOR_RADIUS: f32 = 0.95;

/// Wireframe stroke drawn around the projector box while the crosshair hovers
/// over it, so it reads as clickable.
pub fn projector_outline() -> Vec<LineVertex> {
    let c = Vec3::from_array(PROJECTOR_POS);
    let h = Vec3::new(0.36, 0.30, 0.48);
    let corners: [Vec3; 8] = [
        c + Vec3::new(-h.x, -h.y, -h.z),
        c + Vec3::new(h.x, -h.y, -h.z),
        c + Vec3::new(h.x, h.y, -h.z),
        c + Vec3::new(-h.x, h.y, -h.z),
        c + Vec3::new(-h.x, -h.y, h.z),
        c + Vec3::new(h.x, -h.y, h.z),
        c + Vec3::new(h.x, h.y, h.z),
        c + Vec3::new(-h.x, h.y, h.z),
    ];
    let edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ];
    let mut verts = Vec::with_capacity(edges.len() * 2);
    for (a, b) in edges {
        for i in [a, b] {
            verts.push(LineVertex {
                pos: corners[i].to_array(),
                color: [0.95, 1.0, 0.80, 0.95],
            });
        }
    }
    verts
}

pub fn build_ship() -> (Vec<Vertex>, Vec<u32>) {
    let mut m = MeshBuilder::new();

    let hull = [0.045, 0.048, 0.062, 1.0];
    let rib = [0.130, 0.130, 0.150, 1.0];
    let metal = [0.200, 0.210, 0.240, 1.0];
    let pale = [0.300, 0.310, 0.340, 1.0];
    let red = [0.280, 0.040, 0.060, 1.0];

    // Main spine.
    m.cylinder(
        Vec3::new(0.0, 0.0, -16.0),
        Vec3::new(0.0, 0.0, 4.0),
        3.2,
        3.2,
        72,
        hull,
        false,
    );
    // Nose cone and tip.
    m.cylinder(
        Vec3::new(0.0, 0.0, 4.0),
        Vec3::new(0.0, 0.0, 9.6),
        3.2,
        0.45,
        72,
        hull,
        true,
    );
    m.cylinder(
        Vec3::new(0.0, 0.0, 9.6),
        Vec3::new(0.0, 0.0, 10.6),
        0.45,
        0.02,
        24,
        metal,
        false,
    );
    // Aft section and engine bell.
    m.cylinder(
        Vec3::new(0.0, 0.0, -16.0),
        Vec3::new(0.0, 0.0, -20.0),
        3.2,
        2.1,
        72,
        hull,
        true,
    );
    m.cylinder(
        Vec3::new(0.0, 0.0, -20.0),
        Vec3::new(0.0, 0.0, -21.4),
        2.1,
        3.0,
        72,
        metal,
        false,
    );

    // Structural bands.
    for z in [
        -15.0f32, -13.5, -12.0, -10.5, -9.0, -7.5, -6.0, -4.5, -3.0, -1.5, 0.0, 1.5, 3.0,
    ] {
        m.cylinder(
            Vec3::new(0.0, 0.0, z - 0.07),
            Vec3::new(0.0, 0.0, z + 0.07),
            3.28,
            3.28,
            72,
            rib,
            false,
        );
    }

    // Red accent rings.
    m.cylinder(
        Vec3::new(0.0, 0.0, 1.2),
        Vec3::new(0.0, 0.0, 1.5),
        3.24,
        3.24,
        72,
        red,
        false,
    );
    m.cylinder(
        Vec3::new(0.0, 0.0, -11.5),
        Vec3::new(0.0, 0.0, -11.2),
        3.24,
        3.24,
        72,
        red,
        false,
    );

    // Dorsal rail.
    m.cuboid(
        Vec3::new(0.0, 3.32, -6.0),
        Vec3::new(0.13, 0.10, 9.2),
        metal,
    );

    // Handrails.
    for x in [-1.7f32, 1.7] {
        m.cylinder(
            Vec3::new(x, 3.3, -8.5),
            Vec3::new(x, 3.3, 1.0),
            0.035,
            0.035,
            11,
            metal,
            false,
        );
        for z in [-8.5f32, -3.8, 1.0] {
            m.cuboid(Vec3::new(x, 3.25, z), Vec3::new(0.05, 0.12, 0.05), metal);
        }
    }

    // Film projector on a scissor mast: the switch for the light field.
    let dark = [0.055, 0.055, 0.065, 1.0];
    for x in [-0.16f32, 0.16] {
        m.cylinder(
            Vec3::new(x, 3.30, 1.55),
            Vec3::new(x, 3.95, 2.45),
            0.035,
            0.035,
            9,
            metal,
            false,
        );
        m.cylinder(
            Vec3::new(x, 3.95, 1.55),
            Vec3::new(x, 3.30, 2.45),
            0.035,
            0.035,
            9,
            metal,
            false,
        );
        m.cylinder(
            Vec3::new(x, 3.95, 1.55),
            Vec3::new(x, 4.60, 2.45),
            0.035,
            0.035,
            9,
            metal,
            false,
        );
        m.cylinder(
            Vec3::new(x, 4.60, 1.55),
            Vec3::new(x, 3.95, 2.45),
            0.035,
            0.035,
            9,
            metal,
            false,
        );
    }
    m.cuboid(
        Vec3::new(0.0, 4.64, 2.0),
        Vec3::new(0.24, 0.035, 0.52),
        pale,
    );
    m.cuboid(Vec3::new(0.0, 4.92, 2.0), Vec3::new(0.30, 0.24, 0.42), dark);
    m.cuboid(
        Vec3::new(0.0, 5.18, 1.86),
        Vec3::new(0.17, 0.035, 0.24),
        red,
    );
    m.cylinder(
        Vec3::new(0.0, 4.92, 2.42),
        Vec3::new(0.0, 4.92, 2.78),
        0.15,
        0.12,
        18,
        pale,
        false,
    );

    // Centrifuge ring with spokes.
    m.torus(
        Vec3::new(0.0, 0.0, -12.0),
        Vec3::Z,
        8.4,
        0.22,
        120,
        12,
        metal,
    );
    for (c, h) in [
        (Vec3::new(4.2, 0.0, -12.0), Vec3::new(4.2, 0.09, 0.09)),
        (Vec3::new(-4.2, 0.0, -12.0), Vec3::new(4.2, 0.09, 0.09)),
        (Vec3::new(0.0, 4.2, -12.0), Vec3::new(0.09, 4.2, 0.09)),
        (Vec3::new(0.0, -4.2, -12.0), Vec3::new(0.09, 4.2, 0.09)),
    ] {
        m.cuboid(c, h, metal);
    }

    // Dorsal equipment pod.
    m.cylinder(
        Vec3::new(1.9, 2.2, -9.5),
        Vec3::new(1.9, 2.2, -4.0),
        0.85,
        0.85,
        24,
        pale,
        true,
    );
    m.cylinder(
        Vec3::new(1.9, 2.2, -4.0),
        Vec3::new(1.9, 2.2, -3.3),
        0.85,
        0.35,
        24,
        hull,
        false,
    );

    // Tether anchor block.
    m.cuboid(Vec3::new(0.0, 3.42, -1.0), Vec3::new(0.16, 0.10, 0.16), red);

    // Dorsal equipment and piping.
    m.cuboid(
        Vec3::new(0.0, 3.45, -7.0),
        Vec3::new(0.45, 0.18, 0.80),
        metal,
    );
    m.cuboid(
        Vec3::new(0.9, 3.35, -4.5),
        Vec3::new(0.30, 0.15, 0.50),
        metal,
    );
    m.cuboid(
        Vec3::new(-1.0, 3.40, 0.2),
        Vec3::new(0.22, 0.14, 0.34),
        metal,
    );
    m.cylinder(
        Vec3::new(-1.9, 2.72, -11.0),
        Vec3::new(-1.9, 2.72, -2.0),
        0.07,
        0.07,
        9,
        metal,
        false,
    );
    m.cylinder(
        Vec3::new(2.0, 2.60, -13.0),
        Vec3::new(2.0, 2.60, -8.0),
        0.05,
        0.05,
        9,
        red,
        false,
    );

    (m.verts, m.idx)
}

pub fn build_sky() -> (Vec<Vertex>, Vec<u32>) {
    let mut m = MeshBuilder::new();
    let center = Vec3::new(0.0, 6.0, -1.0);
    let radius = 450.0;
    let seg = 64usize;
    let rings = 32usize;
    let color = [1.0, 1.0, 1.0, 1.0];

    let point = |ring: usize, s: usize| -> Vec3 {
        let phi = std::f32::consts::PI * ring as f32 / rings as f32;
        let theta = TAU * s as f32 / seg as f32;
        center + radius * Vec3::new(phi.sin() * theta.sin(), phi.cos(), phi.sin() * theta.cos())
    };

    for ring in 0..rings {
        for s in 0..seg {
            let p00 = point(ring, s);
            let p10 = point(ring + 1, s);
            let p11 = point(ring + 1, s + 1);
            let p01 = point(ring, s + 1);
            m.quad(p00, p10, p11, p01, Vec3::Y, color);
        }
    }

    (m.verts, m.idx)
}

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed | 1)
    }

    pub fn f(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32) / 16_777_216.0
    }

    pub fn range(&mut self, a: f32, b: f32) -> f32 {
        a + (b - a) * self.f()
    }
}

pub fn build_particles() -> Vec<Particle> {
    let mut rng = Rng::new(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(120_000);

    // The volume must match the wrap box in shader.wgsl (half extents), centred
    // on the camera's starting point, so the field is full in every direction
    // from the first frame.
    const CX: f32 = 0.8;
    const CY: f32 = 6.2;
    const CZ: f32 = -6.2;
    const HX: f32 = 34.0;
    const HY: f32 = 26.0;
    const HZ: f32 = 48.0;

    // Dust: fine grains streaming along the line axis.
    for _ in 0..85_000 {
        let size = rng.range(0.005, 0.024);
        let warm = rng.f();
        let tint = [1.0, 0.38 + 0.11 * warm, 0.42 + 0.11 * warm];
        out.push(Particle {
            center: [
                CX + rng.range(-HX, HX),
                CY + rng.range(-HY, HY),
                CZ + rng.range(-HZ, HZ),
            ],
            size,
            seed: rng.f(),
            speed: rng.range(2.0, 6.0),
            tint,
            _pad: 0.0,
        });
    }

    // Bokeh: out-of-focus discs, evenly spread through the whole volume so they
    // surround the viewer instead of only sitting along the ship's axis.
    for i in 0..4_500 {
        let size = if i % 16 < 1 {
            0.30 + 0.75 * rng.f().powf(1.6)
        } else {
            0.07 + 0.85 * rng.f().powf(2.6)
        };
        let hot = rng.f();
        let dim = 0.50 + 0.55 * rng.f();
        let tint = [dim, dim * (0.26 + 0.24 * hot), dim * (0.32 + 0.24 * hot)];
        out.push(Particle {
            center: [
                CX + rng.range(-HX, HX),
                CY + rng.range(-HY, HY),
                CZ + rng.range(-HZ, HZ),
            ],
            size,
            seed: rng.f(),
            speed: rng.range(2.0, 5.8),
            tint,
            _pad: 0.0,
        });
    }

    // Mid-sized glowing motes between dust and bokeh.
    for _ in 0..52_000 {
        let warm = rng.f();
        out.push(Particle {
            center: [
                CX + rng.range(-HX, HX),
                CY + rng.range(-HY, HY),
                CZ + rng.range(-HZ, HZ),
            ],
            size: rng.range(0.028, 0.088),
            seed: rng.f(),
            speed: rng.range(1.5, 4.5),
            tint: [1.0, 0.40 + 0.10 * warm, 0.45 + 0.10 * warm],
            _pad: 0.0,
        });
    }

    out
}
