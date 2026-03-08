import re
from typing import List

from swift.plugin import ORM, orms
from swift.utils import get_logger
from nltk.translate.bleu_score import sentence_bleu

logger = get_logger()


# Code borrowed from plugin/orm.py
class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            "The math_verify package is required but not installed. Please install it using 'pip install math_verify'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match', extraction_config=[LatexExtractionConfig()])
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # Reward 1 if the content is the same as the ground truth, 0 otherwise
                reward = float(verify(answer_parsed, gold_parsed))
            else:
                # If the gold solution is not parseable, we reward 1 to skip this example
                reward = 1.0
            rewards.append(reward)
        return rewards


class MathFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CountdownORM(ORM):

    def __call__(self, completions, target, nums, **kwargs) -> List[float]:
        """
        Evaluates completions based on Mathematical correctness of the answer

        Args:
            completions (list[str]): Generated outputs
            target (list[str]): Expected answers
            nums (list[str]): Available numbers

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for completion, gt, numbers in zip(completions, target, nums):
            try:
                # Check if the format is correct
                match = re.search(r'<answer>(.*?)<\/answer>', completion)
                if match is None:
                    rewards.append(0.0)
                    continue
                # Extract the "answer" part from the completion
                equation = match.group(1).strip()
                if '=' in equation:
                    equation = equation.split('=')[0]
                # Extract all numbers from the equation
                used_numbers = [int(n) for n in re.findall(r'\d+', equation)]

                # Check if all numbers are used exactly once
                if sorted(used_numbers) != sorted(numbers):
                    rewards.append(0.0)
                    continue
                # Define a regex pattern that only allows numbers, operators, parentheses, and whitespace
                allowed_pattern = r'^[\d+\-*/().\s]+$'
                if not re.match(allowed_pattern, equation):
                    rewards.append(0.0)
                    continue

                # Evaluate the equation with restricted globals and locals
                result = eval(equation, {"__builti'ns__": None}, {})
                # Check if the equation is correct and matches the ground truth
                if abs(float(result) - float(gt)) < 1e-5:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
            except Exception:
                # If evaluation fails, reward is 0
                rewards.append(0.0)
        return rewards


class MultiModalAccuracyORM(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.

        Returns:
            list[float]: Reward scores
        """
        rewards = []
        from math_verify import parse, verify
        for content, sol in zip(completions, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                answer = parse(content)
                if float(verify(answer, parse(sol))) > 0:
                    reward = 1.0
            except Exception:
                pass  # Continue to next verification method if this fails

            # If symbolic verification failed, try string matching
            if reward == 0.0:
                try:
                    # Extract answer from solution if it has think/answer tags
                    sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

                    # Extract answer from content if it has think/answer tags
                    content_match = re.search(r'<answer>(.*?)</answer>', content)
                    student_answer = content_match.group(1).strip() if content_match else content.strip()

                    # Compare the extracted answers
                    if student_answer == ground_truth:
                        reward = 1.0
                except Exception:
                    pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


class radgraphORM(ORM):
    def __init__(self):
        from radgraph import F1RadGraph
        self.rg = F1RadGraph(reward_level="all")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.
        Returns:
            list[float]: Reward scores
        """
        # rewards = []
        try:
        # content_match = [re.search(r'<answer>(.*?)</answer>', content) for content in completions]
        # content_match = [match.group(1).strip() if match else match.strip() for match in content_match]
        #     content_match = [re.search(r'<answer>(.*?)</answer>', match).group(1).strip() if re.search(r'<answer>(.*?)</answer>', match) else match.strip() for match in completions]
            content_match = [match.strip() for match in completions]
            sol_match = solution
            mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = self.rg(
                hyps=content_match, refs=sol_match)
            assert len(reward_list[0]) == len(completions)
            return reward_list[0] * 0.6
        except Exception as e:
            # logger.warning(f"radgraphORM failed: {e}")
            return [0.0] * len(completions)

class hybridNLG_ORM(ORM):
    def __init__(self):
        from bleu.bleu import Bleu
        from meteor.meteor import Meteor
        from rouge.rouge import Rouge
        self.bleu = Bleu()
        self.meteor = Meteor()
        self.rouge = Rouge()

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.
        Returns:
            list[float]: Reward scores
        """
        # rewards = []
        # try:
        # content_match = [re.search(r'<answer>(.*?)</answer>', content) for content in completions]
        # content_match = [match.group(1).strip() if match else match.strip() for match in content_match]
        #     content_match = [re.search(r'<answer>(.*?)</answer>', match).group(1).strip() if re.search(r'<answer>(.*?)</answer>', match) else match.strip() for match in completions]
        predictions = [match.strip() for match in completions]
        labels = solution
        eval_res_bleu = self.bleu.compute(predictions=predictions, references=labels)['bleu']
        eval_res_meteor = self.meteor.compute(predictions=predictions, references=labels)['meteor']
        eval_res_rouge = self.rouge.compute(predictions=predictions, references=labels)['rougeLsum']
        return (eval_res_bleu + eval_res_meteor + eval_res_rouge) / 3.0 * 0.4
        # except Exception as e:
        #     # logger.warning(f"radgraphORM failed: {e}")
        #     return [0.0] * len(completions)

class radgraph_think_ORM(ORM):
    def __init__(self):
        from radgraph import F1RadGraph
        self.rg = F1RadGraph(reward_level="all")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.
        Returns:
            list[float]: Reward scores
        """
        # rewards = []
        try:
        # content_match = [re.search(r'<answer>(.*?)</answer>', content) for content in completions]
        # content_match = [match.group(1).strip() if match else match.strip() for match in content_match]
            content_match = [re.search(r'<think>(.*?)</think>', match).group(1).strip() if re.search(r'<think>(.*?)</think>', match) else match.strip() for match in completions]
            sol_match = solution
            mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = self.rg(
                hyps=content_match, refs=sol_match)
            assert len(reward_list[0]) == len(completions)
            return reward_list[0]
        except Exception as e:
            # logger.warning(f"radgraphORM failed: {e}")
            return [0.0] * len(completions)


class totalAccORM(ORM):
    def __call__(self, completions, solution, **kwargs) -> List[float]:
        """
        Reward function that checks if the completion is correct.
        Args:
            completions (list[str]): Generated outputs
            solution (list[str]): Ground Truths.
        Returns:
            list[float]: Reward scores
        """
        rewards = []
        for content, sol in zip(completions, solution):
            reward = 0.0
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = sol
                # ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                ground_truth = sol_match

                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()

                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
                else:
                    bleu_score = sentence_bleu([ground_truth], student_answer)
                    reward = bleu_score
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
            rewards.append(reward)
        return rewards


orms['external_math_acc'] = MathAccuracy
orms['external_math_format'] = MathFormat
orms['external_countdown'] = CountdownORM
orms['external_r1v_acc'] = MultiModalAccuracyORM
orms['external_radgraph'] = radgraphORM
orms['external_radgraph_think'] = radgraph_think_ORM
orms['external_total_acc'] = totalAccORM
orms['external_hybridNLG'] = hybridNLG_ORM
