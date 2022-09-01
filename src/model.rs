// 유전자 알고리즘 -- 재산 분배 문제
// 2022-09-01
// 
// YeJun Jung (yejun614@naver.com)

use std::thread;
use std::sync::mpsc;
use std::io::{stdout, stdin, Write, Read};
use rand::prelude::{Rng};
// use plotters::prelude::*;
use chrono::{Local, DateTime};

#[derive(Clone)]
pub struct GAModelTracker {
    pub local_datetime: DateTime<Local>,
    pub best_gene: Gene,
    pub total_generation: i32,
    pub best_fitness_changes: Vec<f64>,
    pub average_fitness_changes: Vec<f64>,
    pub average_diff_changes: Vec<f64>,
}

impl Default for GAModelTracker {
    fn default() -> Self {
        Self {
            local_datetime: Local::now(),
            best_gene: Gene { data: Vec::new(), fitness: 1.0 },
            total_generation: 0,
            best_fitness_changes: Vec::<f64>::new(),
            average_fitness_changes: Vec::<f64>::new(),
            average_diff_changes: Vec::<f64>::new(),
        }
    }
}

impl GAModelTracker {
    fn reset(&mut self) {
        self.total_generation = 0;
        self.best_gene = Gene { data: Vec::new(), fitness: 1.0 };
        self.best_fitness_changes.clear();
        self.average_fitness_changes.clear();
        self.average_diff_changes.clear();
    }

    fn append(&mut self, best_fitness: f64, average_fitness: f64, average_diff: f64) {
        self.total_generation += 1;
        self.best_fitness_changes.push(best_fitness);
        self.average_fitness_changes.push(average_fitness);
        self.average_diff_changes.push(average_diff);
    }

    // pub fn get_graph(&self, file_name: &str) {
    //     let max_y = self.average_fitness_changes[0];
    //     let plt = BitMapBackend::new(file_name, (1920, 1080)).into_drawing_area();
    //     plt.fill(&WHITE).unwrap();

    //     let mut chart = ChartBuilder::on(&plt)
    //         .caption(format!("Model Fitness Changes (local_datetime: {:?})", self.local_datetime), ("Arial", 20))
    //         .set_label_area_size(LabelAreaPosition::Left, 80)
    //         .set_label_area_size(LabelAreaPosition::Right, 80)
    //         .set_label_area_size(LabelAreaPosition::Top, 30)
    //         .set_label_area_size(LabelAreaPosition::Bottom, 40)
    //         .margin(10)
    //         .build_cartesian_2d(-100..self.total_generation + 100, -1000..max_y + 1000)
    //         .unwrap();

    //     chart.configure_mesh().draw().unwrap();

    //     chart.draw_series(
    //         LineSeries::new((0..self.total_generation).map(|x| (x, self.best_fitness_changes[x as usize])), &BLUE)
    //     ).unwrap();

    //     chart.draw_series(
    //         LineSeries::new((0..self.total_generation).map(|x| (x, self.average_fitness_changes[x as usize])), &RED)
    //     ).unwrap();

    //     chart.draw_series(
    //         LineSeries::new((0..self.total_generation).map(|x| (x, self.average_diff_changes[x as usize])), &GREEN)
    //     ).unwrap();
    // }
}

#[derive(Clone)]
pub struct GAModel {
    pub genes: Vec<Gene>,
    pub gene_len: usize,
    pub divide: Vec<f64>,
    pub properties: Vec<i32>,
    pub mutation_probability: f64,
    pub mutation_gene_data_len: usize,
    pub elite_conservation_probability: f64,
    pub total_conservation_probability: f64,
    pub tracker: GAModelTracker,
}

impl Default for GAModel {
    fn default() -> Self {
        Self {
            genes: Vec::<Gene>::new(),
            gene_len: 500,
            divide: Vec::<f64>::new(),
            properties: Vec::<i32>::new(),
            mutation_probability: 0.2,
            mutation_gene_data_len: 5,
            elite_conservation_probability: 0.1,
            total_conservation_probability: 0.9,
            tracker: GAModelTracker::default(),
        }
    }
}

impl GAModel {
    pub fn fit_back(&mut self, generations: usize) -> (thread::JoinHandle<()>, mpsc::Receiver<GAModelTracker>) {
        let (tx, rx) = mpsc::channel();
        let mut model = self.clone();

        let handler = thread::spawn(move || {
            model.tracker.reset();
            model.shake();

            for generation in 0..generations {
                model.run_once(generation);

                tx.send(model.tracker.clone()).unwrap();
            }

            println!("Done.");
        });

        (handler, rx)
    }

    pub fn fit(&mut self, generations: usize) {
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

        self.tracker.reset();
        self.shake();

        for generation in 0..generations {
            self.run_once(generation);
        }

        println!("\n ##################   FIT DONE  ##################\n");
    }

    fn run_once(&mut self, generation: usize) {
        self.set_fitnesses();

        let mut average_fitness: f64 = 0.0;
        for n in 0..self.gene_len {
            average_fitness += self.genes[n].fitness as f64;
        }
        average_fitness /= self.gene_len as f64;

        if self.genes[0].fitness < self.tracker.best_gene.fitness {
            let diff = self.genes[0].fitness - self.tracker.best_gene.fitness;
            self.tracker.best_gene = Gene {
                data: self.genes[0].data.to_vec(),
                fitness: self.genes[0].fitness
            };

            println!("[Generation] {}", generation);
            println!(" [Best] fitness: {}, diffence: {}\n", self.tracker.best_gene.fitness, diff);
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
pub struct Gene {
    pub data: Vec<i32>,
    pub fitness: f64,
}

impl Default for Gene {
    fn default() -> Self {
        Self {
            data: Vec::<i32>::new(),
            fitness: 1.0,
        }
    }
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
