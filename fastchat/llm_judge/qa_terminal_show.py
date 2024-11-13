import argparse
from collections import defaultdict
import re

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_single_model_judgments,
    load_pairwise_model_judgments,
    resolve_single_judgment_dict,
    resolve_pairwise_judgment_dict,
    get_single_judge_explanation,
    get_pairwise_judge_explanation,
)

# Initialize dictionaries
questions = []
model_answers = {}
model_judgments_normal_single = {}
model_judgments_math_single = {}
model_judgments_normal_pairwise = {}
model_judgments_math_pairwise = {}

category_selector_map = defaultdict(list)
question_selector_map = {}

newline_pattern1 = re.compile("\n\n(\d+\. )")
newline_pattern2 = re.compile("\n\n(- )")


def post_process_answer(x):
    """Fix Markdown rendering problems."""
    x = x.replace("\u2022", "- ")
    x = re.sub(newline_pattern1, "\n\g<1>", x)
    x = re.sub(newline_pattern2, "\n\g<1>", x)
    return x


def display_question(category):
    choices = category_selector_map[category]
    return choices[0], choices


def display_pairwise_answer(question_id, model_selector1, model_selector2):
    q = question_selector_map[question_id]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    ans2 = model_answers[model_selector2][qid]

    # Generate explanations
    judgment_dict = resolve_pairwise_judgment_dict(
        q, model_judgments_normal_pairwise, model_judgments_math_pairwise, multi_turn=False
    )
    explanation = get_pairwise_judge_explanation((qid, model_selector1, model_selector2), judgment_dict)

    judgment_dict_turn2 = resolve_pairwise_judgment_dict(
        q, model_judgments_normal_pairwise, model_judgments_math_pairwise, multi_turn=True
    )
    explanation_turn2 = get_pairwise_judge_explanation((qid, model_selector1, model_selector2), judgment_dict_turn2)

    print("First Turn Judgment:\n", explanation)
    print("Second Turn Judgment:\n", explanation_turn2)


def display_single_answer(question_id, model_selector):
    q = question_selector_map[question_id]
    qid = q["question_id"]

    ans1 = model_answers[model_selector][qid]
    judgment_dict = resolve_single_judgment_dict(
        q, model_judgments_normal_single, model_judgments_math_single, multi_turn=False
    )
    explanation = get_single_judge_explanation((qid, model_selector), judgment_dict)

    judgment_dict_turn2 = resolve_single_judgment_dict(
        q, model_judgments_normal_single, model_judgments_math_single, multi_turn=True
    )
    explanation_turn2 = get_single_judge_explanation((qid, model_selector), judgment_dict_turn2)

    print("First Turn Judgment:\n", explanation)
    print("Second Turn Judgment:\n", explanation_turn2)


def build_question_selector_map():
    global question_selector_map, category_selector_map
    for q in questions:
        preview = f"{q['question_id']}: {q['turns'][0][:128]}..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


def main(args):
    # Load questions, answers, and judgments
    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    pairwise_model_judgment_file = f"data/{args.bench_name}/model_judgment/gpt-4_pair.jsonl"
    single_model_judgment_file = f"data/{args.bench_name}/model_judgment/gpt-4_single.jsonl"

    global questions, model_answers, model_judgments_normal_single, model_judgments_math_single
    global model_judgments_normal_pairwise, model_judgments_math_pairwise

    questions = load_questions(question_file, None, None)
    model_answers = load_model_answers(answer_dir)
    model_judgments_normal_single = model_judgments_math_single = load_single_model_judgments(single_model_judgment_file)
    model_judgments_normal_pairwise = model_judgments_math_pairwise = load_pairwise_model_judgments(pairwise_model_judgment_file)

    # Build question-selector mappings
    build_question_selector_map()

    if args.mode == "pairwise":
        category = input("Enter the category of questions: ")
        question_id, choices = display_question(category)
        model_selector1 = input(f"Choose Model A from options {list(model_answers.keys())}: ")
        model_selector2 = input(f"Choose Model B from options {list(model_answers.keys())}: ")
        display_pairwise_answer(question_id, model_selector1, model_selector2)

    elif args.mode == "single":
        category = input("Enter the category of questions: ")
        question_id, choices = display_question(category)
        model_selector = input(f"Choose Model from options {list(model_answers.keys())}: ")
        display_single_answer(question_id, model_selector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench-name", type=str, default="mt_bench", help="Name of the benchmark")
    parser.add_argument("--mode", choices=["single", "pairwise"], required=True, help="Select 'single' or 'pairwise' mode for model evaluation.")
    args = parser.parse_args()

    main(args)

