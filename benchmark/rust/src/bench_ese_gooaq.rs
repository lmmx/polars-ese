use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

fn read_texts(path: &str) -> Vec<String> {
    let raw = fs::read_to_string(path).expect("failed to read texts file");
    raw.lines()
        .map(|line| {
            line.replace("\\n", "\n")
                .replace("\\t", "\t")
                .replace("\\\\", "\\")
        })
        .collect()
}

fn read_single(path: &PathBuf) -> String {
    fs::read_to_string(path)
        .expect("failed to read single-text file")
        .replace("\\n", "\n")
        .replace("\\t", "\t")
        .replace("\\\\", "\\")
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let bulk_path = &args[1];
    let sweep_dir = PathBuf::from(&args[2]);
    let batch_sizes: Vec<usize> = args[3]
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();
    let length_targets: Vec<usize> = args[4]
        .split(',')
        .filter_map(|s| s.parse().ok())
        .collect();

    // --- Bulk: cold + hot ---
    let texts = read_texts(bulk_path);

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

    // --- Single-query QPS ---
    let short = "typed dictionaries";
    let long_owned: String = "typed dictionaries and mappings ".repeat(20);
    let long = long_owned.as_str();

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

    // --- Batch sweep ---
    for size in &batch_sizes {
        let batch_path = sweep_dir.join(format!("batch_{size}.tsv"));
        if !batch_path.exists() {
            continue;
        }
        let batch = read_texts(batch_path.to_str().unwrap());
        // warmup
        for _ in 0..3 {
            let _ = ese::encode(&batch);
        }
        let iters = if *size <= 16 {
            2000
        } else if *size <= 1000 {
            500
        } else {
            50
        };
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(ese::encode(std::hint::black_box(&batch)));
        }
        let dt = t0.elapsed().as_secs_f64();
        let rows_s = (iters * size) as f64 / dt;
        println!("batch {size} {rows_s}");
    }

    // --- Length sweep ---
    for char_len in &length_targets {
        let path = sweep_dir.join(format!("len_{char_len}.tsv"));
        if !path.exists() {
            continue;
        }
        let text = read_single(&path);
        for _ in 0..10_000 {
            let _ = ese::encode_single(&text);
        }
        let iters: usize = 200_000;
        let t0 = Instant::now();
        for _ in 0..iters {
            std::hint::black_box(ese::encode_single(std::hint::black_box(&text)));
        }
        let dt = t0.elapsed().as_secs_f64();
        let ops_s = iters as f64 / dt;
        let us_op = (dt * 1e6) / iters as f64;
        println!("len {char_len} {ops_s} {us_op}");
    }
}
