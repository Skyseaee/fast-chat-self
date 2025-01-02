# generate the pairwise-all result for all model, support deepseek api ( default ), support local API

import argparse
from ast import List
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import concurrent
import openai
import shortuuid
import numpy as np
from sympy import Li
from tqdm import tqdm

from fastchat.llm_judge.gen_api_answer import get_answer, reorg_answer_file
from fastchat.llm_judge.common import ( 
        load_questions, 
        load_model_answers, 
        load_judge_prompts, 
        get_model_list, 
        play_a_match_single, 
        run_judge_pair, 
        play_a_match_pair, 
        check_data,
        NEED_REF_CATS,
)
from fastchat.llm_judge.gen_judgement import make_judge_single, make_judge_pairwise, make_match_single, make_match_all_pairs, make_match
from fastchat.llm_judge.show_result import (
    display_result_single,
    display_result_pairwise,
    display_result_pairwise_single
)

def gen_answer_file_name(bench_name: str, model_id: List[str]) -> List[str]:
    filenames = []
    for model in model_id:
        answer_file = f"data/{bench_name}/model_answer/{model}.jsonl"
        filenames.append(answer_file)
    
    return filenames

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bench-name",
        type=str,
        default="mt_bench",
        help="The name of the benchmark question set.",
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--model-list",
        type=str,
        nargs="+",
        default=None,
        help="A list of models to be evaluated",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--force-temperature", type=float, help="Forcibly set a sampling temperature."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--openai-api-base", 
        type=str,
        nargs='+', 
        default=None
    )
    parser.add_argument(
        "--local-api",
        action="store_true",
        help="Use the local API instead of the default OpenAI or DeepSeek API. ",
    )

    # args for gen judgements
    parser.add_argument(
        "--judge-file",
        type=str,
        default="data/judge_prompts.jsonl",
        help="The file of judge prompts.",
    )

    parser.add_argument("--judge-model", type=str, default="gpt-4")
    parser.add_argument("--baseline-model", type=str, default="gpt-3.5-turbo")
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["pairwise-baseline", "pairwise-all", "single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`single` runs single answer grading."
        ),
    )

    parser.add_argument(
        "--parallel", type=int, default=1, help="The number of concurrent API calls."
    )
    parser.add_argument(
        "--first-n", type=int, help="A debug option. Only run the first `n` judgments."
    )


    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )

    # args for show result
    parser.add_argument(
        "--result-mode",
        type=str,
        default="pairwise-single",
        choices=["pairwise-baseline", "pairwise-all", "single", "pairwise-single"],
        help=(
            "Evaluation mode. "
            "`pairwise-baseline` runs pairwise comparision against a baseline. "
            "`pairwise-all` runs pairwise comparision between all pairs. "
            "`pairwise-single` runs a single pairwise comparison for all possible pairs. "
            "`single` runs single answer grading."
        ),
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"data/{args.bench_name}/question.jsonl"
    if args.answer_file:
        answer_files = args.answer_file
    else:
        answer_files = gen_answer_file_name(args.bench_name, args.model_list)

    print(f"Output to {answer_files}")

    # if args.openai_api_base is not None:
    #     openai.api_base = args.openai_api_base
    #     openai.api_key = os.environ.get("OPEN_API_KEY")

    question_file = f"data/{args.bench_name}/question.jsonl"
    questions = load_questions(question_file, args.question_begin, args.question_end)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as executor:
        futures = []
        for question in questions:
            for (answer_file, open_api) in zip(answer_files, args.openai_api_base):
                future = executor.submit(
                    get_answer,
                    question,
                    args.model,
                    args.num_choices,
                    args.max_tokens,
                    answer_file,
                    args.local_api,
                    open_api,
                )
                futures.append(future)

        for future in tqdm.tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            future.result()
    reorg_answer_file(answer_files)
    
    answer_dir = f"data/{args.bench_name}/model_answer"
    ref_answer_dir = f"data/{args.bench_name}/reference_answer"

    # Load answers
    model_answers = load_model_answers(answer_dir)
    ref_answers = load_model_answers(ref_answer_dir)

    # Load judge
    judge_prompts = load_judge_prompts(args.judge_file)

    if args.first_n:
        questions = questions[: args.first_n]

    if args.model_list is None:
        models = get_model_list(answer_dir)
    else:
        models = args.model_list

    if args.mode == "single":
        judges = make_judge_single(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        output_file = (
            f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        )
        make_match_func = make_match_single
        baseline_model = None
    else:
        judges = make_judge_pairwise(args.judge_model, judge_prompts)
        play_a_match_func = play_a_match_pair

        if '//' not in args.judge_model:
            output_file = (
                f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
            )
        else:
            judge_model = args.judge_model.split('@')[1]
            output_file = (
                f"data/{args.bench_name}/model_judgment/{judge_model}_pair.jsonl"
            )
        
        if args.mode == "pairwise-all":
            make_match_func = make_match_all_pairs
            baseline_model = None
        else:
            make_match_func = make_match
            baseline_model = args.baseline_model

    check_data(questions, model_answers, ref_answers, models, judges)

    question_math = [q for q in questions if q["category"] in NEED_REF_CATS]
    question_default = [q for q in questions if q["category"] not in NEED_REF_CATS]

    # Make matches
    matches = []
    matches += make_match_func(
        question_default, models, model_answers, judges["default"], baseline_model
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math"],
        baseline_model,
        ref_answers,
    )
    matches += make_match_func(
        question_default,
        models,
        model_answers,
        judges["default-mt"],
        baseline_model,
        multi_turn=True,
    )
    matches += make_match_func(
        question_math,
        models,
        model_answers,
        judges["math-mt"],
        baseline_model,
        ref_answers,
        multi_turn=True,
    )

    match_stat = {}
    match_stat["bench_name"] = args.bench_name
    match_stat["mode"] = args.mode
    match_stat["judge"] = args.judge_model
    match_stat["baseline"] = baseline_model
    match_stat["model_list"] = models
    match_stat["total_num_questions"] = len(questions)
    match_stat["total_num_matches"] = len(matches)
    match_stat["output_path"] = output_file
    
    pair_uuid = shortuuid.uuid()
    args.uuid = pair_uuid
    # Show match stats and prompt enter to continue
    print("Stats:")
    print(json.dumps(match_stat, indent=4))

    input("Press Enter to confirm...")

    # Play matches
    if args.parallel == 1:
        for match in tqdm(matches):
            play_a_match_func(match, output_file=output_file, pair_uuid="")
    else:

        def play_a_match_wrapper(match):
            play_a_match_func(match, output_file=output_file, pair_uuid="")

        np.random.seed(0)
        np.random.shuffle(matches)

        with ThreadPoolExecutor(args.parallel) as executor:
            for match in tqdm(
                executor.map(play_a_match_wrapper, matches), total=len(matches)
            ):
                pass


    # show result
    print('Generate final results.')
    if args.result_mode == "single":
        display_result_func = display_result_single
    elif args.result_mode == "pairwise-single":
        display_result_func = display_result_pairwise_single
    else:
        if args.result_mode == "pairwise-all":
            args.baseline_model = None
        display_result_func = display_result_pairwise

    print(f"Mode: {args.mode}")
    display_result_func(args)

