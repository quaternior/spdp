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

# WandB 초기화

# 로깅 설정 (stdout을 파일로 저장)
log_file = "wanda.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format="%(message)s")

# 최종 목표 sparsity (사용자 설정)

# pruning_1.sh, pruning_2.sh 실행할 conda 환경
env_pruning_1 = "wanda"
env_pruning_2 = "teal"

# 결과 저장 리스트


wandb.init(project="wanda-monitor", name=f"wanda_run")
# 5% 간격으로 s1을 생성 (0% 및 final_s 제외)
for final_s in range (5, 96, 5):
    results = []
    final_s = final_s / 100.0
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"\n[INFO {timestamp}] 현재 sparsity: s={final_s:.2f}"
    print(msg)
    logging.info(msg)
    model_name = f"$SPDP_HOME/wanda/out/llama2_7b/unstructured/wanda{final_s}"
    # pruning_2.sh 실행 (teal 환경)
    print(f"[INFO {timestamp}] teal 환경에서 pruning_2.sh 실행 중...")
    cmd_pruning_2 = f"conda run -n {env_pruning_2} bash $SPDP_HOME/TEAL/scripts/ppl_test_wanda.bash {final_s}"

    process = subprocess.Popen(cmd_pruning_2, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)

    stderr_output = ""
    stdout_output = ""
    for line in iter(process.stdout.readline, ""):
        sys.stdout.write(f"[teal] {line.strip()}\n")
        sys.stdout.flush()
        logging.info(f"[teal] {line.strip()}")
        stdout_output += line  # stdout 저장

    for line in iter(process.stderr.readline, ""):
        sys.stderr.write(f"[teal ERROR] {line.strip()}\n")
        sys.stderr.flush()
        logging.error(f"[teal ERROR] {line.strip()}")
        stderr_output += line

    process.wait()
    if process.returncode != 0:
        print(f"[ERROR {timestamp}] pruning_2.sh 실행 실패 (s={final_s})")
        continue

    # uniform PPL 추출
    pattern = r"Evaluating dense PPL\s+PPL:\s+([\d\.]+)"
    match = re.search(pattern, stderr_output + stdout_output)
    if match:
        ppl_val = float(match.group(1))
        print(f"[INFO {timestamp}] uniform PPL={ppl_val:.3f}")
        logging.info(f"[INFO] uniform PPL={ppl_val:.3f}")
        results.append((final_s, ppl_val))
        wandb.log({"final_s":final_s, "uniform_ppl": ppl_val})
    else:
        print(f"[WARNING {timestamp}] uniform PPL을 찾지 못함 (s={final_s})")

    # # 생성된 모델 디렉토리 삭제 (pickle 파일 등 정리)
    # model_name = f"/root/wanda/out/llama2_7b/unstructured/wanda{s1_float}"
    # if os.path.exists(model_name):
    #     subprocess.run(f"rm -rf {model_name}", shell=True)
    #     print(f"[INFO {timestamp}] {model_name} 디렉토리를 삭제 완료.")
    # else:
    #     print(f"[INFO {timestamp}] {model_name} 디렉토리가 존재하지 않음.")

# CSV 저장
os.makedirs("results", exist_ok=True)
csv_path = f"results/wanda_ppl_results.csv"

df = pd.DataFrame(results, columns=["s","Uniform_PPL"])
df.to_csv(csv_path, index=False)

timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

print(f"[INFO {timestamp}] CSV 파일이 {csv_path} 에 저장되었습니다.")
wandb.save(csv_path)

# TODO
# grabnppl.bash 분리, fine-grained 확인