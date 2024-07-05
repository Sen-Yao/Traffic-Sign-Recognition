# Traffic Sign Recognition

This is a NUS SoC Summer Workshop 2024 Visual Computing Project worked on by Group 10.

Traffic-sign recognition (TSR) is a technology by which a vehicle is able to recognize traffic signs on the road e.g. "speed limit" or "children" or "turn ahead". This is a very important technology in self-driving cars.

This project will give students the chance to train different models using various features to classify traffic signs. The project is divided into three difficulty levels: Beginner, Expert, and Bonus.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Beginner Level](#beginner-level)
- [Bonus Level](#bonus-level)
- [Data](#data)
- [Dependencies](#dependencies)
- [Authors](#authors)
- [License](#license)

## Introduction

Traffic Sign Recognition is a crucial component of autonomous driving systems. This project involves training machine learning models to recognize various traffic signs using the Chinese Traffic Sign Database (CTSD).

## Installation

1. Clone the repository:
```shell
git clone https://github.com/Sen-Yao/traffic-sign-recognition.git
cd traffic-sign-recognition  
```

2. Set up a virtual environment (optional but recommended):

```shell
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install the required dependencies:

```shell
pip install -r requirements.txt
```

## Usage

To run the project, use the following command:

```shell
python main.py
```

The project is divided into 3 difficulty levels. Beginner, Expert and Bonus.

## Beginner Level

For the Beginner level, use the Chinese Traffic Sign Database (http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html). Place the images in the Dataset/CTSD directory.

Extract the HOG feature and use SVM as the classifier.

Example command:


```CLI
python main.py --dataset_name CTSD --feature_extractor hog --classifier svm
```

## Bonus Level

For the Bonus level, use the German Traffic Sign Recognition Benchmark (GTSRB) from here[https://benchmark.ini.rub.de/]. Place the Train, Test, Meta folder and files in the Dataset/GTSRB directory.

## Authors

- Lin Ziyao 
- Li Yihan
- Ma Jianfa
- Zhu Yuchen

## Data

Download the Chinese Traffic Sign Database from here[https://nlpr.ia.ac.cn/pal/trafficdata/recognition.html]. Ensure the images are placed in the Dataset/CTSD directory as required.

Download the German Traffic Sign Recognition Benchmark (GTSRB) from here[https://benchmark.ini.rub.de/]. Ensure the images are placed in the Dataset/GTSRB directory as required.

## Dependencies

List of dependencies can be found in the requirements.txt file. Some key dependencies include:

- OpenCV 
- Scikit-learn 
- Scikit-image 
- NumPy
- tqdm

Install all dependencies using:

```shell
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License. See the LICENSE file for details.
