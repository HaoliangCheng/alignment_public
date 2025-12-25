#!/usr/bin/env python3
"""
Script to transform train.jsonl into sft.jsonl with <think> and <answer> tags.
This script processes the MATH dataset and formats it for supervised fine-tuning.
"""

import json
import argparse
import re
from typing import Optional


def extract_boxed_answer(solution: str) -> Optional[str]:
    """Extract the answer from inside a LaTeX \\boxed{} command"""
    # Find the last occurrence of \boxed
    idx = solution.rfind("\\boxed")
    if idx < 0:
        idx = solution.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(solution):
        if solution[i] == "{":
            num_left_braces_open += 1
        if solution[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    
    boxed_string = solution[idx : right_brace_idx + 1]
    
    # Remove the \boxed{ and } parts
    left = "\\boxed{"
    try:
        assert boxed_string[: len(left)] == left
        assert boxed_string[-1] == "}"
        return boxed_string[len(left) : -1]
    except:
        return None


def extract_final_answer_from_solution(solution: str) -> Optional[str]:
    """
    Extract the final answer from a solution text.
    First tries to extract from \boxed{} commands, then falls back to other methods.
    """
    # Try to extract from \boxed{} first
    boxed_answer = extract_boxed_answer(solution)
    if boxed_answer is not None:
        return boxed_answer
    
    # Fallback: look for common answer patterns
    # Pattern 1: "The answer is X" or "Therefore, X" at the end
    patterns = [
        r"[Tt]he answer is\s+(.+?)\.?\s*$",
        r"[Tt]herefore,?\s+(.+?)\.?\s*$",
        r"[Ss]o,?\s+(.+?)\.?\s*$",
        r"[Hh]ence,?\s+(.+?)\.?\s*$",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, solution.strip(), re.MULTILINE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Pattern 2: Look for mathematical expressions at the end
    lines = solution.strip().split('\n')
    if lines:
        last_line = lines[-1].strip()
        # If the last line looks like a mathematical expression or number
        if re.match(r'^[0-9\-\+\*/\^\(\)\{\}\\\$\s\.]+$', last_line):
            return last_line
    
    # If no clear answer pattern found, return None
    return None


def format_solution_with_tags(problem: str, solution: str) -> str:
    """
    Format the solution with <think> and <answer> tags.
    """
    # Extract the final answer
    final_answer = extract_final_answer_from_solution(solution)
    
    if final_answer is None:
        # If we can't extract the answer, use the last sentence as the answer
        sentences = solution.strip().split('.')
        if len(sentences) > 1:
            final_answer = sentences[-1].strip()
            if not final_answer:
                final_answer = sentences[-2].strip() if len(sentences) > 2 else solution.strip()
        else:
            final_answer = solution.strip()
    
    # Format with think and answer tags
    formatted_response = f"<think>\n{solution.strip()}\n</think> <answer>{final_answer}</answer>"
    
    return formatted_response


def transform_jsonl(input_file: str, output_file: str, include_problem_in_response: bool = False):
    """
    Transform the input JSONL file to SFT format.
    
    Args:
        input_file: Path to input train.jsonl file
        output_file: Path to output sft.jsonl file
        include_problem_in_response: Whether to include the problem statement in the response
    """
    print(f"Reading from {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        processed_count = 0
        failed_count = 0
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the JSON line
                data = json.loads(line.strip())
                
                # Extract problem and solution
                problem = data.get('problem', '')
                solution = data.get('solution', '')
                
                if not problem or not solution:
                    print(f"Warning: Missing problem or solution at line {line_num}")
                    failed_count += 1
                    continue
                
                # Create the SFT format
                if include_problem_in_response:
                    # Include problem in the response (less common)
                    formatted_response = f"Problem: {problem}\n\n" + format_solution_with_tags(problem, solution)
                    sft_data = {
                        "prompt": "",  # Empty prompt since problem is in response
                        "response": formatted_response,
                        **{k: v for k, v in data.items() if k not in ['problem', 'solution']}
                    }
                else:
                    # More common format: problem as prompt, solution as response
                    sft_data = {
                        "prompt": problem,
                        "response": format_solution_with_tags(problem, solution),
                        **{k: v for k, v in data.items() if k not in ['problem', 'solution']}
                    }
                
                # Write to output file
                outfile.write(json.dumps(sft_data, ensure_ascii=False) + '\n')
                processed_count += 1
                
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} examples...")
                    
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {line_num}: {e}")
                failed_count += 1
            except Exception as e:
                print(f"Error processing line {line_num}: {e}")
                failed_count += 1
    
    print(f"\nTransformation complete!")
    print(f"Successfully processed: {processed_count} examples")
    print(f"Failed to process: {failed_count} examples")
    print(f"Output saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Transform train.jsonl to sft.jsonl with think/answer tags")
    parser.add_argument("--input", "-i", default="data/MATH/train.jsonl", 
                       help="Input train.jsonl file path (default: data/MATH/train.jsonl)")
    parser.add_argument("--output", "-o", default="sft.jsonl", 
                       help="Output sft.jsonl file path (default: sft.jsonl)")
    parser.add_argument("--include-problem", action="store_true",
                       help="Include problem statement in the response (default: use problem as prompt)")
    
    args = parser.parse_args()
    
    try:
        transform_jsonl(args.input, args.output, args.include_problem)
    except FileNotFoundError:
        print(f"Error: Input file {args.input} not found!")
        print("Available train files:")
        import os
        for root, dirs, files in os.walk("data"):
            for file in files:
                if file == "train.jsonl":
                    print(f"  {os.path.join(root, file)}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main() 