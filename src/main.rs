// 유전자 알고리즘 -- 재산 분배 문제
// 2022-08-31
// 
// YeJun Jung (yejun614@naver.com)

use std::fs;
use std::time::Instant;
use std::io::{stdout, stdin, Write, Read};
use rand::prelude::{Rng};
use plotters::prelude::*;
use chrono::{Local, DateTime};

struct GAModelTracker {
    local_datetime: DateTime<Local>,
    total_generation: i32,
    best_fitness_changes: Vec<i32>,
    average_fitness_changes: Vec<i32>,
    average_diff_changes: Vec<i32>,
}

impl GAModelTracker {
    fn reset(&mut self) {
        self.total_generation = 0;
        self.best_fitness_changes.clear();
        self.average_fitness_changes.clear();
    }

    fn append(&mut self, best_fitness: f64, average_fitness: f64, average_diff: f64) {
        self.total_generation += 1;
        self.best_fitness_changes.push((best_fitness * 100000000.0) as i32);
        self.average_fitness_changes.push((average_fitness * 100000000.0) as i32);
        self.average_diff_changes.push((average_diff * 100000000.0) as i32);
    }

    fn get_graph(&self, file_name: &str) {
        let max_y = self.average_fitness_changes[0];
        let plt = BitMapBackend::new(file_name, (1920, 1080)).into_drawing_area();
        plt.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&plt)
            .caption(format!("Model Fitness Changes (local_datetime: {:?})", self.local_datetime), ("Arial", 20))
            .set_label_area_size(LabelAreaPosition::Left, 80)
            .set_label_area_size(LabelAreaPosition::Right, 80)
            .set_label_area_size(LabelAreaPosition::Top, 30)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .margin(10)
            .build_cartesian_2d(-100..self.total_generation + 100, -1000..max_y + 1000)
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(
            LineSeries::new((0..self.total_generation).map(|x| (x, self.best_fitness_changes[x as usize])), &BLUE)
        ).unwrap();

        chart.draw_series(
            LineSeries::new((0..self.total_generation).map(|x| (x, self.average_fitness_changes[x as usize])), &RED)
        ).unwrap();

        chart.draw_series(
            LineSeries::new((0..self.total_generation).map(|x| (x, self.average_diff_changes[x as usize])), &GREEN)
        ).unwrap();
    }
}

struct GAModel {
    genes: Vec<Gene>,
    gene_len: usize,
    divide: Vec<f64>,
    properties: Vec<i32>,
    mutation_probability: f64,
    mutation_gene_data_len: usize,
    elite_conservation_probability: f64,
    total_conservation_probability: f64,
    tracker: GAModelTracker,
}

impl GAModel {
    fn fit(&mut self, generations: usize) -> Gene {
        let mut best_gene = Gene { data: Vec::new(), fitness: 1.0 };
        self.tracker.reset();

        println!("\n ##################  PROPERTIES ##################\n");

        println!(" [local_datetime] {:?}", self.tracker.local_datetime);
        print!("\n");

        println!(" [gene_len] {}", self.gene_len);
        println!(" [mutataion_properbability] {}", self.mutation_probability);
        println!(" [mutation_gene_data_len] {}", self.mutation_gene_data_len);
        println!(" [elite_conservation_probability] {}", self.elite_conservation_probability);
        println!(" [total_conservation_probability] {}", self.total_conservation_probability);
        print!("\n");

        println!(" [divide] {:?}", self.divide);
        println!(" [properties] {:?}", self.properties);

        println!("\n Press enter key to continue ...");
        stdout().flush().unwrap();
        stdin().read(&mut [0]).unwrap();

        println!("\n ##################  FIT START  ##################\n");

        self.shake();

        for generation in 0..generations {

            self.set_fitnesses();

            let mut average_fitness: f64 = 0.0;
            for n in 0..self.gene_len {
                average_fitness += self.genes[n].fitness as f64;
            }
            average_fitness /= self.gene_len as f64;

            if self.genes[0].fitness < best_gene.fitness {
                let diff = self.genes[0].fitness - best_gene.fitness;
                best_gene = Gene {
                    data: self.genes[0].data.to_vec(),
                    fitness: self.genes[0].fitness
                };

                println!("[Generation] {}", generation);
                println!(" [Best] fitness: {}, diffence: {}\n", best_gene.fitness, diff);
            }

            self.selection();
            
            self.mutation();

            let mut total_average_diff: f64 = 0.0;

            for y in 0..self.gene_len {
                let mut average_diff: f64 = 0.0;

                for x in 0..self.gene_len {
                    average_diff += (self.genes[y].fitness - self.genes[x].fitness).abs();
                }

                average_diff /= self.gene_len as f64;
                total_average_diff += average_diff;
            }

            total_average_diff /= self.gene_len as f64;
            self.tracker.append(self.genes[0].fitness, average_fitness, total_average_diff);
        }

        println!("\n ##################   FIT DONE  ##################\n");

        best_gene
    }

    fn shake(&mut self) {
        let max_value: i32 = self.divide.len() as i32;
        let gene_data_len: usize = self.properties.len();

        self.genes.clear();

        for _n in 0..self.gene_len {
            let mut new_gene = Gene { data: Vec::<i32>::new(), fitness: 1.0 };
            new_gene.init(gene_data_len, 0, max_value);

            self.genes.push(new_gene);
        }
    }

    fn set_fitnesses(&mut self) {

        for n in 0..self.gene_len {
            self.genes[n].set_fitness(&self.divide, &self.properties)
        }

        self.genes.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());
    }

    fn selection(&mut self) {
        let mut rng = rand::thread_rng();
        let mut child = Vec::<Gene>::new();

        let mut elite_len: usize = ((self.gene_len as f64) * self.elite_conservation_probability) as usize;
        let total_len: usize = ((self.gene_len as f64) * self.total_conservation_probability) as usize;

        if elite_len % 2 != 0 {
            elite_len -= 1;
        }

        // Elite conservation
        for n in 0..elite_len {
            let new_gene = Gene {
                data: self.genes[n].data.to_vec(),
                fitness: self.genes[n].fitness
            };

            child.push(new_gene);
        }

        // Gene crossover
        while child.len() < self.gene_len {
            let n1 = rng.gen_range(0..total_len);
            let n2 = rng.gen_range(0..total_len);

            let result = self.genes[n1].uniform_crossover(&self.genes[n2]);

            child.push(result.0);
            child.push(result.1);
        }

        // Swap generation
        self.genes = (&child[0..self.gene_len]).to_vec();

        if child.len() != self.gene_len {
            println!("  [ERROR] child length is wrong. ({})", child.len());
        }
    }

    fn mutation(&mut self) {
        let mut rng = rand::thread_rng();

        let max_value: i32 = self.divide.len() as i32;
        let elite_len: usize = ((self.gene_len as f64) * self.elite_conservation_probability) as usize;
        let mutation_len: i32 = ((self.gene_len as f64) * self.mutation_probability) as i32;

        for _n in 0..mutation_len {
            let index = rng.gen_range(elite_len..self.gene_len);
            self.genes[index] = self.genes[index].mutation(self.mutation_gene_data_len, 0, max_value);
        }
    }
}

#[derive(Clone)]
struct Gene {
    data: Vec<i32>,
    fitness: f64,
}

impl Gene {
    fn init(&mut self, data_len: usize, min_value: i32, max_value: i32) {
        let mut rng = rand::thread_rng();

        self.data.clear();
        for _n in 0..data_len {
            self.data.push(rng.gen_range(min_value..max_value));
        }
    }

    fn set_fitness(&mut self, divide: &[f64], properties: &[i32]) {
        let len = self.data.len();
        let data_num = divide.len();
        let properties_sum: i32 = properties.iter().sum();

        let mut data_divide: Vec<i32> = vec![0; data_num];

        for n in 0..len {
            data_divide[self.data[n] as usize] += properties[n];
        }

        self.fitness = 0.0;

        for n in 0..data_num {
            let ratio: f64 = (data_divide[n] as f64) / (properties_sum as f64);
            self.fitness += (divide[n] - ratio).abs();
        }
    }

    fn crossover(&self, another: &Gene) -> (Gene, Gene) {
        let len = self.data.len();
        let mut rng = rand::thread_rng();

        let p1 = rng.gen_range(0..len);
        let p2 = rng.gen_range(0..len);

        let mut g1 = Gene { data: vec![0; len], fitness: -1.0 };
        let mut g2 = Gene { data: vec![0; len], fitness: -1.0 };

        for n in 0..len {
            if n >= p1 && n <= p2 {
                g1.data[n] = self.data[n];
                g2.data[n] = another.data[n];
            } else {
                g1.data[n] = another.data[n];
                g2.data[n] = self.data[n];
            }
        }

        (g1, g2)
    }

    fn uniform_crossover(&self, another: &Gene) -> (Gene, Gene) {
        let len = self.data.len();
        let mut rng = rand::thread_rng();

        let mut mask = vec![false; len];
        for n in 0..len {
            let dice: f64 = rng.gen();
            mask[n] = dice > 0.5;
        }

        let mut g1 = Gene { data: vec![0; len], fitness: -1.0 };
        let mut g2 = Gene { data: vec![0; len], fitness: -1.0 };

        for n in 0..len {
            if mask[n] {
                g1.data[n] = another.data[n];
                g2.data[n] = self.data[n];
            } else {
                g1.data[n] = self.data[n];
                g2.data[n] = another.data[n];
            }
        }

        (g1, g2)
    }

    fn average_crossover(&self, another: &Gene) -> Gene {
        let len = self.data.len();

        let mut new_gene = Gene { data: vec![0; len], fitness: -1.0 };

        for n in 0..len {
            new_gene.data[n] = (self.data[n] + another.data[n]) / 2;
        }

        new_gene
    }

    fn mutation(&self, count: usize, min_value: i32, max_value: i32) -> Gene {
        let len = self.data.len();
        let mut rng = rand::thread_rng();

        let mut gene = Gene {
            data: self.data.clone(),
            fitness: 0.0,
        };

        for _n in 0..count {
            let index = rng.gen_range(0..len);
            gene.data[index] = rng.gen_range(min_value..max_value);
        }

        gene
    }

    fn compare(&self, another: &Gene, data_num: usize) -> i32 {
        let len = self.data.len();
        let mut data_count: Vec<i32> = vec![0; data_num];

        for n in 0..len {
            data_count[self.data[n] as usize] += 1;
            data_count[another.data[n] as usize] -= 1;
        }

        let mut diff: i32 = 0;

        for n in 0..data_num {
            diff += data_count[n].abs();
        }

        diff
    }
}

fn load_dataset(divide_path: &str, properties_path: &str) -> (Vec<f64>, Vec<i32>) {
    let divide_contents = fs::read_to_string(divide_path).expect(" [ERROR] File read error");
    let properties_contents = fs::read_to_string(properties_path).expect(" [ERROR] File read error");

    let mut divide = Vec::<f64>::new();

    for val in divide_contents.split(" ") {
        divide.push(val.parse::<f64>().unwrap());
    }

    let mut properties = Vec::<i32>::new();

    for val in properties_contents.split(" ") {
        properties.push(val.parse::<i32>().unwrap());
    }

    (divide, properties)
}

fn main() {
    // Welcome
    println!("\nGenetic Algorithm Example with Rust");
    println!(" -- Worth distribution problem");
    println!(" -- YeJun, Jung (yejun614@naver.com)");
    println!(" -- 2022-08-31");
    print!("\n");

    // Read datasets
    let mut divide_file_path = String::new();
    let mut properties_file_path = String::new();

    print!(" | Divide File Path (10) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut divide_file_path).unwrap();
    divide_file_path = divide_file_path.trim().to_string();

    if divide_file_path.is_empty() {
        divide_file_path = "10".to_string();
    }

    divide_file_path = format!("./property/divide{}.txt", divide_file_path);

    print!(" | Properties File Path (100) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut properties_file_path).unwrap();
    properties_file_path = properties_file_path.trim().to_string();

    if properties_file_path.is_empty() {
        properties_file_path = "100".to_string();
    }

    properties_file_path = format!("./property/properties{}.txt", properties_file_path);

    let (divide, properties) = load_dataset(&divide_file_path.trim(), &properties_file_path.trim());
    println!(" [Done] Dataset is loaded.\n");

    // Create genetic algorithm model
    let mut model = GAModel {
        genes: Vec::<Gene>::new(),
        gene_len: 500,
        divide: divide,
        properties: properties,
        mutation_probability: 0.2,
        mutation_gene_data_len: 5,
        elite_conservation_probability: 0.1,
        total_conservation_probability: 0.9,
        tracker: GAModelTracker {
            local_datetime: Local::now(),
            total_generation: 0,
            best_fitness_changes: Vec::<i32>::new(),
            average_fitness_changes: Vec::<i32>::new(),
            average_diff_changes: Vec::<i32>::new(),
        }
    };

    // Input model properties
    let mut input_buf = String::new();

    print!(" | model.gene_len (500) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if !input_buf.is_empty() {
        model.gene_len = input_buf.parse::<usize>().unwrap();
    }

    input_buf = String::new();

    print!(" | model.mutation_probability (0.2) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if !input_buf.is_empty() {
        model.mutation_probability = input_buf.parse::<f64>().unwrap();
    }

    input_buf = String::new();

    print!(" | model.mutation_gene_data_len (5) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if !input_buf.is_empty() {
        model.mutation_gene_data_len = input_buf.parse::<usize>().unwrap();
    }

    input_buf = String::new();

    print!(" | model.elite_conservation_probability (0.1) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if !input_buf.is_empty() {
        model.elite_conservation_probability = input_buf.parse::<f64>().unwrap();
    }

    input_buf = String::new();

    print!(" | model.total_conservation_probability (0.9) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if !input_buf.is_empty() {
        model.total_conservation_probability = input_buf.parse::<f64>().unwrap();
    }

    input_buf = String::new();

    print!(" | generation (5000) > ");
    stdout().flush().unwrap();
    stdin().read_line(&mut input_buf).unwrap();
    input_buf = input_buf.trim().to_string();

    if input_buf.is_empty() {
        input_buf = "5000".to_string();
    }
    let generation = input_buf.parse::<usize>().unwrap();

    // Start fitting
    let now = Instant::now();
    let best_gene = model.fit(generation);
    let elapsed = now.elapsed();

    // Print fit results
    println!("[Fit results]");
    println!(" [Elapsed] {:?}", elapsed);

    println!(" [Gene data]");
    println!("{:?}\n", best_gene.data);

    println!(" [Best fitness] {}", best_gene.fitness);

    let properties_sum: i32 = model.properties.iter().sum();
    println!(" [Real fitness] {}",  (properties_sum as f64) * best_gene.fitness);

    // Export fitness plots
    model.tracker.get_graph("fitness_chagnes.png");
}
