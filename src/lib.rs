// 유전자 알고리즘 -- 재산 분배 문제
// 2022-09-01
// 
// YeJun Jung (yejun614@naver.com)

use std::fs;
use std::io::{stdout, stdin, Write};

pub mod model;

pub mod app;

pub fn load_dataset(divide_path: &str, properties_path: &str) -> (Vec<f64>, Vec<i32>) {
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

pub fn read_line_with_default<T>(message: &str, read: &mut T, default: T) where
    T: Clone + std::fmt::Display + std::str::FromStr
{
    print!(" | {} ({}) > ", message, default.to_string());

    let mut read_str = String::new();

    stdout().flush().unwrap();
    stdin().read_line(&mut read_str).unwrap();
    read_str = read_str.trim().to_string();

    if read_str.is_empty() {
        *read = default.clone();
    } else {
        match read_str.parse::<T>() {
            Ok(value) => *read = value,
            Err(_e) => panic!("read_str parse error!"),
        }
    }
}

