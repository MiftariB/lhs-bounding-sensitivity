configfile: "snakemake_config.yaml"
import itertools

BOUND_TYPES = config["BOUND_TYPES"]
PROBLEMS = config["PROBLEMS"]
NS = config["NS"]
UPPER_BOUNDS_SELECTED = config["UPPER_BOUNDS_SELECTED"]
LOWER_BOUNDS_SELECTED = config["LOWER_BOUNDS_SELECTED"]
output_dir = "output"

rule all:
    input:
        expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json", bound=BOUND_TYPES.keys(), problem=PROBLEMS, n=NS, type=["ub", "lb"]),
        expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.png", bound=BOUND_TYPES.keys(),problem=PROBLEMS, n=NS, type=["ub", "lb"]),
        expand(output_dir + "/truth/{problem}/{n}.json", bound=BOUND_TYPES.keys(), problem=PROBLEMS, n=[100]),
        expand(output_dir + "/it_refining/{problem}/{bnd[0]}-{bnd[1]}.json", problem=PROBLEMS, bnd=itertools.combinations(BOUND_TYPES, 2)),
        expand(output_dir + "/it_refining/{problem}/{bnd[0]}-{bnd[1]}.mp4", problem=PROBLEMS, bnd=itertools.combinations(BOUND_TYPES, 2)),
        expand(output_dir + "/it_refining_solo/{problem}/{bnd[0]}-{bnd[1]}.json", problem=PROBLEMS, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED)),
        expand(output_dir + "/it_refining_solo/{problem}/{bnd[0]}-{bnd[1]}.mp4", problem=PROBLEMS, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED))

rule std_bounds:
    threads: 5
    output: output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json"
    log: "logs/bounds/{type}/{bound}/{problem}/{n}.log"
    wildcard_constraints:
        bound="|".join([x for x, y in BOUND_TYPES.items() if y])
    shell: "python experiments_std_bounds.py {wildcards.type} {wildcards.n} {wildcards.problem} {wildcards.bound} 5400 {output} &> {log}"

rule truth:
    threads: 5
    output:
        output_dir + "/truth/{problem}/{n}.json",
        output_dir + "/truth/{problem}/{n}.npz"
    log: "logs/truth/{problem}/{n}.log"
    shell: "python experiments_truth.py {wildcards.n} {wildcards.problem} {output[0]} {output[1]} &> {log}"

rule bound_img:
    input:
        output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json",
        output_dir + "/truth/{problem}/100.npz"
    output: output_dir + "/bounds/{type}/{bound}/{problem}/{n}.png"
    notebook: "gen_bound_plot.ipynb"

rule gen_table:
    input:
        expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json", bound=BOUND_TYPES.keys(), problem=PROBLEMS, n=NS, type=["ub", "lb"]),
    output:
        output_dir + "/table.json"
    notebook: "gen_table.ipynb"

rule it_refining:
    threads: 5
    output:
        output_dir + "/it_refining/{problem}/{bnd1}-{bnd2}.json"
    log: "logs/it_refining/{problem}/{bnd1}-{bnd2}.log"
    shell: "python experiments_it_refining.py {wildcards.problem} 180 {output} {wildcards.bnd1} {wildcards.bnd2} &> {log}"

rule it_refining_mp4:
    threads: 1
    input:
        output_dir + "/truth/{problem}/100.npz",
        output_dir + "/it_refining/{problem}/{bnd1}-{bnd2}.json"
    output:
        output_dir + "/it_refining/{problem}/{bnd1}-{bnd2}.mp4"
    log: "logs/it_refining/{problem}/{bnd1}-{bnd2}-mp4.log"
    shell: "python experiments_gen_video.py {input[0]} {input[1]} {output} &> {log}"

rule it_refining_solo:
    threads: 5
    output:
        output_dir + "/it_refining_solo/{problem}/{bnd1}-{bnd2}.json"
    log: "logs/it_refining_solo/{problem}/{bnd1}-{bnd2}.log"
    shell: "python experiments_it_refining.py {wildcards.problem} 180 {output} --solo {wildcards.bnd1} {wildcards.bnd2} &> {log}"

rule it_refining_mp4_solo:
    threads: 1
    input:
        output_dir + "/truth/{problem}/100.npz",
        output_dir + "/it_refining_solo/{problem}/{bnd1}-{bnd2}.json"
    output:
        output_dir + "/it_refining_solo/{problem}/{bnd1}-{bnd2}.mp4"
    log: "logs/it_refining_solo/{problem}/{bnd1}-{bnd2}-mp4.log"
    shell: "python experiments_gen_video.py {input[0]} {input[1]} {output} &> {log}"