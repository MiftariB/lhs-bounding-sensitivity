configfile: "snakemake_config.yaml"
import itertools

BOUND_TYPES_LP = config["BOUND_TYPES_LP"]
BOUND_TYPES_MIP = config["BOUND_TYPES_MIP"]
PROBLEMS_LP = config["LP"]
PROBLEMS_MIP = config["MIP"]
NS = config["NS"]
UPPER_BOUNDS_SELECTED = config["UPPER_BOUNDS_SELECTED"]
LOWER_BOUNDS_SELECTED = config["LOWER_BOUNDS_SELECTED"]
output_dir = "output"

def gen_all_mip_params():
    o_problems = []
    o_n = []
    o_bounds = []
    o_type = []

    for problem in PROBLEMS_MIP:
        for n in NS:
            for bound, btype in BOUND_TYPES_MIP.items():
                for b in btype:
                    o_problems.append(problem)
                    o_n.append(n)
                    o_bounds.append(bound)
                    o_type.append(b)

    todo_mip_refining = []
    available_lbs = ["component_flat"] #[bound for bound, btypes in BOUND_TYPES_MIP.items() if "lb" in btypes]
    available_ubs = ["component_flat"] #[bound for bound, btypes in BOUND_TYPES_MIP.items() if "ub" in btypes]

    for lb in available_lbs:
        for ub in available_ubs:
            for problem in PROBLEMS_MIP:
                todo_mip_refining.append((problem, lb, ub))

    return o_problems, o_n, o_bounds, o_type, todo_mip_refining

o_problems, o_n, o_bounds, o_type, todo_mip_refining = gen_all_mip_params()

rule all:
    input:
        #expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json", bound=BOUND_TYPES_LP, problem=PROBLEMS_LP, n=NS, type=["ub", "lb"]),
        #expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.png", bound=BOUND_TYPES_LP,problem=PROBLEMS_LP, n=NS, type=["ub", "lb"]),
        #expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json", zip, bound=o_bounds, problem=o_problems, n=o_n, type=o_type),
        #expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.png", zip, bound=o_bounds,problem=o_problems, n=o_n, type=o_type),
        expand(output_dir + "/truth/{problem}/{n}.json", problem=PROBLEMS_LP + PROBLEMS_MIP, n=[100]),
        expand(output_dir + "/truth/{problem}/{n}.png", problem=PROBLEMS_LP + PROBLEMS_MIP, n=[100]),
        expand(output_dir + "/it_refining/{problem}/lb={bnd[0]}-ub={bnd[1]}.json", problem=PROBLEMS_LP, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED)),
        expand(output_dir + "/it_refining/{problem}/lb={bnd[0]}-ub={bnd[1]}.mp4", problem=PROBLEMS_LP, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED)),
        expand(output_dir + "/it_refining/{todo[0]}/lb={todo[1]}-ub={todo[2]}.json", todo=todo_mip_refining),
        expand(output_dir + "/it_refining/{todo[0]}/lb={todo[1]}-ub={todo[2]}.mp4", todo=todo_mip_refining),
        #expand(output_dir + "/it_refining_solo/{problem}/{bnd[0]}-{bnd[1]}.json", problem=PROBLEMS, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED)),
        #expand(output_dir + "/it_refining_solo/{problem}/{bnd[0]}-{bnd[1]}.mp4", problem=PROBLEMS, bnd=itertools.product(LOWER_BOUNDS_SELECTED, UPPER_BOUNDS_SELECTED))
        #expand(output_dir + "/pareto/{problem}/{type}-{n}.png", problem=PROBLEMS, n=NS, type=["ub", "lb"]),
        #expand("output_first_batch/table.json"),

rule std_bounds:
    priority: 1
    threads: 5
    output: output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json"
    log: "logs/bounds/{type}/{bound}/{problem}/{n}.log"
    wildcard_constraints:
        bound="|".join(BOUND_TYPES_LP)
    shell: "python experiments_std_bounds.py {wildcards.type} {wildcards.n} {wildcards.problem} {wildcards.bound} 5400 {output} &> {log}"

rule truth:
    priority: 2
    threads: 5
    output:
        output_dir + "/truth/{problem}/{n}.json",
        output_dir + "/truth/{problem}/{n}.npz"
    log: "logs/truth/{problem}/{n}.log"
    shell: "python experiments_truth.py {wildcards.n} {wildcards.problem} {output[0]} {output[1]} &> {log}"

rule truth_img:
    priority: 3
    input:
        output_dir + "/truth/{problem}/{n}.npz"
    output: output_dir + "/truth/{problem}/{n}.png"
    notebook: "gen_bound_plot.ipynb"

rule bound_img:
    priority: 4
    input:
        output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json",
        output_dir + "/truth/{problem}/100.npz"
    output: output_dir + "/bounds/{type}/{bound}/{problem}/{n}.png"
    notebook: "gen_bound_plot.ipynb"

rule gen_table:
    input:
        expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json", bound=BOUND_TYPES_LP, problem=PROBLEMS_LP, n=NS, type=["ub", "lb"]),
    output:
        output_dir + "/table.json"
    notebook: "gen_table.ipynb"

rule pareto_front:
    input:
        bounds=expand(output_dir + "/bounds/{type}/{bound}/{problem}/{n}.json",bound=BOUND_TYPES_LP,allow_missing=True),
        truth= output_dir + "/truth/{problem}/100.npz"
    output:
        output_dir + "/pareto/{problem}/{type}-{n}.png"
    notebook: "pareto_front.ipynb"

rule pareto_front_test:
    input: output_dir + "/pareto/toy_1/lb-6.png"

rule it_refining:
    threads: 5
    input:
        truth=output_dir + "/truth/{problem}/100.json"
    output:
        output_dir + "/it_refining/{problem}/lb={bnd1}-ub={bnd2}.json"
    log: "logs/it_refining/{problem}/lb={bnd1}-ub={bnd2}.log"
    shell: "python experiments_it_refining.py {wildcards.problem} 3600 {input.truth} 250 {output} --solo {wildcards.bnd1} {wildcards.bnd2} &> {log}"

rule it_refining_mp4:
    threads: 1
    priority: 5
    input:
        output_dir + "/truth/{problem}/100.npz",
        output_dir + "/it_refining/{problem}/lb={bnd1}-ub={bnd2}.json"
    output:
        output_dir + "/it_refining/{problem}/lb={bnd1}-ub={bnd2}.mp4"
    log: "logs/it_refining/{problem}/lb={bnd1}-ub={bnd2}-mp4.log"
    shell: "python experiments_gen_video.py {input[0]} {input[1]} {output} &> {log}"
