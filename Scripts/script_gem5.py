import subprocess
import sys
import json
import os
import re
import logging
import ast
import time
from tqdm import tqdm
import filecmp

def files_equal_ignore_empty_lines(file1, file2):
    def lines_no_empty(path):
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return [line.strip() for line in f if line.strip() != '']
    return lines_no_empty(file1) == lines_no_empty(file2)

# imported from https://github.com/LearningOpt/pie/blob/main/gem5/benchmarking.py
def calc_sim_seconds(stats):
    return float(stats["simTicks"]) / float(stats["simFreq"]) # more accurate than sim_seconds

def parse_stats_txt(stats_path):
    with open(stats_path, 'r') as f:
        stats_lines = f.readlines()
    
    stats = {}
    for line in stats_lines:
        if line.strip() == '':
            continue
        if "Begin" in line:
            continue
        if "End" in line:
            continue
        line = re.sub("#.*", "", line).strip() # remove comments
        parts = line.split()
        parts = [part.strip() for part in parts]
        if len(parts) > 2: 
            value = parts[1:]
        elif len(parts) == 2:
            value = parts[1]
        else: 
            logging.warn(f'could not parse line {line}')
            continue
        key = parts[0]
        if isinstance(value, str): 
            try: 
                value = value.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None")
                value = ast.literal_eval(value) if value != "None" else None
            except:
                logging.warn(f"could not parse value {value} for key {key}")
        elif isinstance(value, list):
            try: 
                value = [v.replace("%", "").replace("nan", "None").replace("inf", "None").replace("-inf", "None") for v in value]
                value = [ast.literal_eval(v) if v != "None" else None for v in value]
            except:
                logging.warn(f"could not parse value {value} for key {key}")
        stats[key] = value
    stats["sim_seconds_precise"] = calc_sim_seconds(stats)
    return stats

def run_command(command, description):
    result = subprocess.run(command, shell=True)
    return result

with open("comp_code_opt_200ep.json", "r", encoding="utf-8") as file: #PASSARE QUI IL JSON OTTENUTO DA TEST MODEL
    dati = json.load(file)

output = open("gem5_results_200ep.csv", "w") #CAMBIARE CON LE EPOCHS E METTERCI DIFFBASED PER DISAMBIGUARE
output.write("problem,correct,speedup,percentOptimized\n")

for indice, dato in enumerate(dati):
    r=[]
    p=[]
    lista_file = []
    for file in os.listdir("merged_testcases/merged_test_cases/" + dato["problem"]):
        if "input" not in file:
                continue
        lista_file.append(file)
    inputfile_failed_on_input = []
    for test in ["input", "prediction"]:
        with open ("code.cpp", "w") as file:
            file.write(dato[test])

        # 1. Compilazione C++
        run_command(
            "g++ -std=c++17 -O3 -w code.cpp -o code.out 2> results/logs/compile.log",
            "Compilazione C++"
        )

        check_compile = True
        with open ("results/logs/compile.log", "r") as compile_file:
            l = list(compile_file.readlines())
        
            if len(l) > 1:
                print(l)
                check_compile = False

        if not check_compile:
            for _ in lista_file:
                p.append((0, False))
            with open ("results/logs/compile.log", "w") as compile_file:
                compile_file.write("")
            continue
        
        for i, file in enumerate(tqdm(lista_file)):
            if i >= 10:
                break
            if file in inputfile_failed_on_input:
                continue
            print(file)
            startexecution = time.time()
            # 2. Simulazione gem5
            checkStatus = run_command(
                    f"timeout 120s build/X86/gem5.opt -q --outdir=results --stats-file=stats.txt --silent-redirect -r --stdout-file=logs/gem5_stdout.log --stderr-file=logs/gem5_stderr.log simulate.py code.out {"merged_testcases/merged_test_cases/" + dato["problem"] + "/" +file}"
                ,
                "Simulazione gem5"
            )
            endexecution = time.time()-startexecution
            if (endexecution >= 120 and test == "input") or checkStatus.returncode != 0:
                inputfile_failed_on_input.append(file)
                continue

            if test=="input":
                r.append(parse_stats_txt("results/stats.txt")["sim_seconds_precise"])
            else:
                probnr = file.split(".")[1]
                flag = files_equal_ignore_empty_lines("merged_testcases/merged_test_cases/" + dato["problem"] + "/output." + probnr + ".txt", "results/sim_program_stdout.txt")

                print(flag)
                if not flag:
                    print("Output corretto")
                    with open ("merged_testcases/merged_test_cases/" + dato["problem"] + "/output." + probnr + ".txt") as file:
                        for line in file.readlines():
                            print(line)
                    print("Output predetto")
                    with open ("results/sim_program_stdout.txt") as file:
                        for line in file.readlines():
                            print(line)
                p.append((parse_stats_txt("results/stats.txt")["sim_seconds_precise"], flag))

    sommaspeedup = 0
    flag = True
    print("analisi dato", indice)
    print(r)
    print(p)
    for a, (b, c) in zip(r,p):
        if not c:
            sommaspeedup += 1
            flag = False
            continue
        if a > b:
            sommaspeedup += a/b
        else:
            sommaspeedup += 1
    sommaspeedup /= len(r)
    output.write(dato["problem"] + "," + str(int(flag)) + "," + str(sommaspeedup) + "," + ("1" if flag and sommaspeedup >= 1.1 else "0") +"\n" )
output.close()