struct Post {
    time: vec4<f32>,
    res: vec4<f32>,
};

@group(0) @binding(0) var scene_tex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var<uniform> post: Post;

struct VOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var p = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out: VOut;
    out.pos = vec4<f32>(p[i], 0.0, 1.0);
    out.uv = vec2<f32>(p[i].x * 0.5 + 0.5, 0.5 - p[i].y * 0.5);
    return out;
}

fn hash21(p: vec2<f32>) -> f32 {
    var p3 = fract(vec3<f32>(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

fn vnoise(p: vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let u = f * f * (3.0 - 2.0 * f);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn aces(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c2 = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (c2 * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs(in: VOut) -> @location(0) vec4<f32> {
    let t = post.time.x;
    let pj = post.time.y;
    let res = post.res.xy;
    let texel = vec2<f32>(1.0) / res;
    let aspect = res.x / res.y;

    // Slow water-on-glass smear, like the film's wet glass rig.
    let n1 = vnoise(in.uv * vec2<f32>(5.5 * aspect, 5.5) + vec2<f32>(t * 0.045, -t * 0.03));
    let n2 = vnoise(in.uv * vec2<f32>(4.1 * aspect, 6.3) + vec2<f32>(-t * 0.035, t * 0.05));
    let warp = (vec2<f32>(n1, n2) - vec2<f32>(0.5)) * 0.012;
    let uv = clamp(in.uv + warp, vec2<f32>(0.001), vec2<f32>(0.999));

    // Tiny chromatic aberration.
    let ca = (in.uv - vec2<f32>(0.5)) * 0.004;
    let base = textureSample(scene_tex, samp, uv);
    let rr = textureSample(scene_tex, samp, clamp(uv + ca, vec2<f32>(0.001), vec2<f32>(0.999))).r;
    let bb = textureSample(scene_tex, samp, clamp(uv - ca, vec2<f32>(0.001), vec2<f32>(0.999))).b;
    var col = vec3<f32>(rr, base.g, bb);

    // Cheap wide glow from bright pixels.
    var glow = vec3<f32>(0.0);
    let d0 = vec2<f32>(1.0, 0.0);
    let d1 = vec2<f32>(0.7071, 0.7071);
    let d2 = vec2<f32>(0.0, 1.0);
    let d3 = vec2<f32>(-0.7071, 0.7071);
    let d4 = vec2<f32>(-1.0, 0.0);
    let d5 = vec2<f32>(-0.7071, -0.7071);
    let d6 = vec2<f32>(0.0, -1.0);
    let d7 = vec2<f32>(0.7071, -0.7071);

    let thr = vec3<f32>(0.42);
    let r1 = 5.0;
    let r2 = 13.0;
    let r3 = 28.0;
    glow += max(textureSample(scene_tex, samp, uv + d0 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d1 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d2 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d3 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d4 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d5 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d6 * texel * r1).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d7 * texel * r1).rgb - thr, vec3<f32>(0.0));

    glow += max(textureSample(scene_tex, samp, uv + d0 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d1 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d2 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d3 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d4 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d5 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d6 * texel * r2).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d7 * texel * r2).rgb - thr, vec3<f32>(0.0));

    glow += max(textureSample(scene_tex, samp, uv + d0 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d1 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d2 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d3 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d4 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d5 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d6 * texel * r3).rgb - thr, vec3<f32>(0.0));
    glow += max(textureSample(scene_tex, samp, uv + d7 * texel * r3).rgb - thr, vec3<f32>(0.0));

    glow *= 0.052;
    col += glow * mix(vec3<f32>(0.70, 1.0, 0.65), vec3<f32>(1.0, 0.46, 0.55), pj) * mix(0.55, 1.0, pj);
    let veil = vnoise(in.uv * vec2<f32>(2.5 * aspect, 2.5) + vec2<f32>(t * 0.02, t * 0.015));
    col += vec3<f32>(0.022, 0.004, 0.008) * (0.80 + 0.5 * veil) * pj;

    // Hot pink grade: bright, saturated highlights with a warm lift, while
    // deep space stays dark. Only while the projector runs.
    let lum = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    col *= 1.0 + 0.55 * pj * smoothstep(0.012, 0.20, lum);
    col += vec3<f32>(0.115, 0.038, 0.048) * pj * smoothstep(0.08, 0.75, lum);
    let lm = dot(col, vec3<f32>(0.2126, 0.7152, 0.0722));
    col = mix(vec3<f32>(lm), col, mix(1.06, 1.18, pj));

    col *= mix(1.0, 1.1, pj);
    var outc = aces(col);

    let vigd = length((in.uv - vec2<f32>(0.5)) * vec2<f32>(aspect, 1.0));
    outc *= 1.0 - 0.55 * vigd * vigd;

    let grain = (hash21(in.uv * res + fract(t) * 911.0) - 0.5) * 0.025;
    outc += grain * (0.6 + 0.4 * (1.0 - outc.b));

    return vec4<f32>(outc, 1.0);
}
