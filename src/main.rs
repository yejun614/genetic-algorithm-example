// 유전자 알고리즘 -- 재산 분배 문제
// 2022-09-01
// 
// YeJun Jung (yejun614@naver.com)

use std::env;
use std::time::Instant;

use rust_genetic_algorithm::*;
use rust_genetic_algorithm::app::GeneApp;
use rust_genetic_algorithm::model::GAModel;

fn main() {
    // Get console arguments
    let args: Vec<String> = env::args().collect();

    if args.contains(&String::from("--console")) {
        // CLI MODE
        run_cli();
    } else {
        // GUI MODE
        run_gui();
    }
}

fn run_gui() {
    let app = GeneApp::default();
    app.run_native();
}

fn run_cli() {
        // Welcome
    println!("\nGenetic Algorithm Example with Rust");
    println!(" -- Property distribution problem");
    println!(" -- YeJun, Jung (yejun614@naver.com)");
    println!(" -- 2022-08-31");
    print!("\n");

    // Read datasets
    let mut divide_file_path = String::new();
    let mut properties_file_path = String::new();

    read_line_with_default("Divide File Path", &mut divide_file_path, "10".to_string());
    divide_file_path = format!("./property/divide{}.txt", divide_file_path);

    read_line_with_default("Properties File Path", &mut properties_file_path, "100".to_string());
    properties_file_path = format!("./property/properties{}.txt", properties_file_path);

    let (divide, properties) = load_dataset(&divide_file_path, &properties_file_path);
    println!(" [Done] Dataset is loaded.\n");

    // Create genetic algorithm model
    let mut model = GAModel::default();

    // Set datasets
    model.divide = divide;
    model.properties = properties;

    // Input model properties
    read_line_with_default("model.gene_len", &mut model.gene_len, 500 as usize);
    read_line_with_default("model.mutation_probability", &mut model.mutation_probability, 0.2);
    read_line_with_default("model.mutation_gene_data_len", &mut model.mutation_gene_data_len, 5 as usize);
    read_line_with_default("model.elite_conservation_probability", &mut model.elite_conservation_probability, 0.1);
    read_line_with_default("model.total_conservation_probability", &mut model.total_conservation_probability, 0.9);

    let mut generation = 0 as usize;
    read_line_with_default("generation", &mut generation, 5000 as usize);

    // Start fitting
    let now = Instant::now();
    model.fit(generation);
    let elapsed = now.elapsed();

    // Print fit results
    println!("[Fit results]");
    println!(" [Elapsed] {:?}", elapsed);

    println!(" [Gene data]");
    println!("{:?}\n", model.tracker.best_gene.data);

    println!(" [Best fitness] {}", model.tracker.best_gene.fitness);

    let properties_sum: i32 = model.properties.iter().sum();
    println!(" [Real fitness] {}",  (properties_sum as f64) * model.tracker.best_gene.fitness);

    // Export fitness plots
    // model.tracker.get_graph("fitness_chagnes.png");
}
