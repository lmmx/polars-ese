use std::env;
use std::fs;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();
    let path = &args[1];

    let raw = fs::read_to_string(path).expect("failed to read texts file");
    let texts: Vec<String> = raw
        .lines()
        .map(|line| {
            line.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
        })
        .collect();

    // --- Bulk: cold + hot on the full corpus ---
    let t0 = Instant::now();
    let _ = ese::encode(&texts);
    let cold = t0.elapsed().as_secs_f64();

    for _ in 0..2 {
        let _ = ese::encode(&texts);
    }

    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t0 = Instant::now();
        let _ = ese::encode(&texts);
        let t = t0.elapsed().as_secs_f64();
        if t < best {
            best = t;
        }
    }

    // --- Single-query QPS (matches the flowercomputers blog methodology) ---
    let short = "typed dictionaries";
    let long_owned: String = "typed dictionaries and mappings ".repeat(20);
    let long = long_owned.as_str();

    // Warm
    for _ in 0..10_000 {
        let _ = ese::encode_single(short);
    }

    let iters: usize = 1_000_000;

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ese::encode_single(std::hint::black_box(short)));
    }
    let short_qps = iters as f64 / t0.elapsed().as_secs_f64();

    for _ in 0..1_000 {
        let _ = ese::encode_single(long);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ese::encode_single(std::hint::black_box(long)));
    }
    let long_qps = iters as f64 / t0.elapsed().as_secs_f64();

    println!("cold {cold}");
    println!("hot {best}");
    println!("dim {}", ese::DIMENSIONS);
    println!("short_qps {short_qps}");
    println!("long_qps {long_qps}");
}
