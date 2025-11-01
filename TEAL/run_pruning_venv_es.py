#!/usr/bin/env python
# coding: utf-8

import subprocess
import re
import os
import sys
import numpy as np
import pandas as pd
import wandb
import logging
from datetime import datetime

# 사용할 GPU index
gpu_idx = 0

# 로깅 설정 (stdout을 파일로 저장)
log_file = f"pruning_{gpu_idx}.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")

# 최종 목표 sparsity (사용자 설정)

# pruning_1.sh, pruning_2.sh 실행할 conda 환경
env_pruning_1 = "$SPDP_HOME/prune_llm"
env_pruning_2 = "$SPDP_HOME/teal"

method = "wanda"

eval_sparsity = True

venv_sh = "bin/activate"

import argparse
parser = argparse.ArgumentParser(description="Parse command line arguments for the script.")
parser.add_argument('--eval_target', type=str, choices=["teal", "spdp", "both"], required=True, help='Evaluate sparsity of teal or spdp')
args = parser.parse_args()

def run_teal():
    for s2 in range(5, 81, 5):
        s2_float = s2 / 100.0
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"\n[INFO {timestamp}] 현재 sparsity: s1={s2_float:.2f}"

        print(msg)
        logging.info(msg)

        # pruning_1.sh 실행 (wanda 환경)
        model_name = f"meta-llama/Llama-2-7b-hf"
        print(model_name)
        # pruning_2.sh 실행 (teal 환경)
        print(f"[INFO {timestamp}] teal 환경에서 ppl_test_teal.bash 실행 중...")
        cmd_pruning_2 = f"source {env_pruning_1}/{venv_sh} && source $SPDP_HOME/TEAL/scripts/ppl_test_teal.bash {s2_float} {eval_sparsity}"

        process = subprocess.Popen(
            cmd_pruning_2, 
            shell=True, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1
        )

        stderr_output = ""
        stdout_output = ""
        for line in iter(process.stdout.readline, ""):
            if not line:
                break
            sys.stdout.write(f"[teal] {line.strip()}\n")
            sys.stdout.flush()
            logging.info(f"[teal] {line.strip()}")
            stdout_output += line

        process.wait()
        if process.returncode != 0:
            print(f"[ERROR {timestamp}] pruning_2.sh 실행 실패 (s2={s2_float})")
            return

def run_spdp():
    for final_s in range (5, 81, 5):
        results = []
        final_s = final_s / 100.0
        optimal_s1_list=[0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.25, 0.25, 0.3, 0.35, 0.4, 0.45]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[INFO {timestamp}] 현재 sparsity: s={final_s}")
        wandb.init(project="pruning-monitor", name=f"pruning_run_{final_s:.2f}_{method}")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"\n[INFO {timestamp}] 현재 sparsity: s={final_s:.2f}"

        logging.info(msg)
        for s1 in optimal_s1_list:
            s1_float = s1 / 100.0
            s2_float = 1.0 - (1.0 - final_s) / (1.0 - s1_float)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            msg = f"\n[INFO {timestamp}] 현재 sparsity: s1={s1_float:.2f}, s2={s2_float:.2f}"
            
            print(msg)
            logging.info(msg)
            wandb.log({"s1": s1_float, "s2": s2_float})

            # pruning_1.sh 실행 (wanda 환경)
            model_name = f"{os.environ.get("SPDP_HOME")}/wanda/out/llama2_7b/unstructured/{method}{s1_float}"
            print(model_name)
            if not os.path.exists(model_name):
                print(f"[INFO {timestamp}] wanda({method}) 바탕 wanda.sh 실행 중...")
                cmd_pruning_1 = f"source {env_pruning_1}/{venv_sh} && source $SPDP_HOME/wanda/wanda.sh {final_s} {s1_float} {method} {gpu_idx}"

                process = subprocess.Popen(
                    cmd_pruning_1,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,  # stderr → stdout에 합치기
                    text=True,
                    bufsize=1
                )

                stdout_output = ""
                for line in iter(process.stdout.readline, ""):
                    if not line:
                        break
                    sys.stdout.write(f"[prune_llm] {line.strip()}\n")
                    sys.stdout.flush()
                    logging.info(f"[prune_llm] {line.strip()}")
                    stdout_output += line

                process.wait()

                if process.returncode != 0:
                    print(f"[ERROR {timestamp}] pruning_1.sh 실행 실패 (s1={s1_float})")
                    continue
            else:
                print(f"[INFO {timestamp}] wanda({method}) 생략")
            # pruning_2.sh 실행 (teal 환경)
            print(f"[INFO {timestamp}] teal 환경에서 grabnppl.bash 실행 중...")
            cmd_pruning_2 = f"source {env_pruning_1}/{venv_sh} && source $SPDP_HOME/TEAL/scripts/grabnppl.bash {final_s} {s1_float} {s2_float} {method} {gpu_idx} {eval_sparsity}"

            process = subprocess.Popen(
                cmd_pruning_2, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1
            )

            stderr_output = ""
            stdout_output = ""
            for line in iter(process.stdout.readline, ""):
                if not line:
                    break
                sys.stdout.write(f"[teal] {line.strip()}\n")
                sys.stdout.flush()
                logging.info(f"[teal] {line.strip()}")
                stdout_output += line

            process.wait()
            if process.returncode != 0:
                print(f"[ERROR {timestamp}] pruning_2.sh 실행 실패 (s2={s2_float})")
                continue

            # uniform PPL 추출
            pattern = r"Evaluating uniform PPL\s+PPL:\s+([\d\.]+)"
            match = re.search(pattern, stderr_output + stdout_output)
            if match:
                ppl_val = float(match.group(1))
                print(f"[INFO {timestamp}] s1={s1_float:.2f}, s2={s2_float:.2f} → uniform PPL={ppl_val:.3f}")
                logging.info(f"[INFO] s1={s1_float:.2f}, s2={s2_float:.2f} → uniform PPL={ppl_val:.3f}")
                results.append((s1_float, s2_float, ppl_val))
                wandb.log({"s1": s1_float, "s2": s2_float, "uniform_ppl": ppl_val})
            else:
                print(f"[WARNING {timestamp}] uniform PPL을 찾지 못함 (s1={s1_float})")

            # # 생성된 모델 디렉토리 삭제 (pickle 파일 등 정리)
            # model_name = f"/root/wanda/out/llama2_7b/unstructured/wanda{s1_float}"
            # if os.path.exists(model_name):
            #     subprocess.run(f"rm -rf {model_name}", shell=True)
            #     print(f"[INFO {timestamp}] {model_name} 디렉토리를 삭제 완료.")
            # else:
            #     print(f"[INFO {timestamp}] {model_name} 디렉토리가 존재하지 않음.")

        # CSV 저장
        os.makedirs("results", exist_ok=True)
        csv_path = f"results/uniform_ppl_results_{method}_{final_s}.csv"

        df = pd.DataFrame(results, columns=["s1", "s2", "Uniform_PPL"])
        df.to_csv(csv_path, index=False)

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print(f"[INFO {timestamp}] CSV 파일이 {csv_path} 에 저장되었습니다.")
        wandb.save(csv_path)

# 5% 간격으로 s1을 생성 (0% 및 final_s 제외)
if args.eval_target == "teal":
    run_teal()
    
elif args.eval_target == "spdp":
    run_spdp()

else:
    run_teal()
    run_spdp()