from __future__ import annotations

import json
import os
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from skillopt.datasets.base import BatchSpec
from skillopt.envs.base import EnvAdapter
from skillopt.envs.seneca_compliance.loader import SenecaComplianceLoader
from skillopt.model import chat_target
from skillopt.gradient.reflect import run_minibatch_reflect

# Import the Seneca single file validator directly by adding the SOP directory to sys.path
_SOP_DIR = "D:/Seneca/01_Engine_SOP"
if _SOP_DIR not in sys.path:
    sys.path.insert(0, _SOP_DIR)

try:
    import skillopt_grader_single as grader_lib
except ImportError:
    grader_lib = None


class SenecaComplianceAdapter(EnvAdapter):
    def __init__(
        self,
        split_dir: str = "",
        data_path: str = "",
        split_mode: str = "split_dir",
        split_ratio: str = "2:1:7",
        split_seed: int = 42,
        split_output_dir: str = "",
        workers: int = 4,
        analyst_workers: int = 4,
        failure_only: bool = False,
        minibatch_size: int = 8,
        edit_budget: int = 4,
        seed: int = 42,
        limit: int = 0,
        max_completion_tokens: int = 4096,
    ) -> None:
        self.workers = workers
        self.analyst_workers = analyst_workers
        self.failure_only = failure_only
        self.minibatch_size = minibatch_size
        self.edit_budget = edit_budget
        self.max_completion_tokens = int(max_completion_tokens)
        self.dataloader = SenecaComplianceLoader(
            split_dir=split_dir,
            data_path=data_path,
            split_mode=split_mode,
            split_ratio=split_ratio,
            split_seed=split_seed,
            split_output_dir=split_output_dir,
            seed=seed,
            limit=limit,
        )

    def setup(self, cfg: dict) -> None:
        super().setup(cfg)
        self.dataloader.setup(cfg)

    def get_dataloader(self):
        return self.dataloader

    def build_env_from_batch(self, batch: BatchSpec, **kwargs):
        return list(batch.payload or [])

    def build_train_env(self, batch_size: int, seed: int, **kwargs):
        batch = self.dataloader.build_train_batch(
            batch_size=batch_size, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def build_eval_env(self, env_num: int, split: str, seed: int, **kwargs):
        batch = self.dataloader.build_eval_batch(
            env_num=env_num, split=split, seed=seed, **kwargs
        )
        return self.build_env_from_batch(batch, **kwargs)

    def _rollout_one(self, item: dict, skill_content: str, out_dir: str) -> dict:
        item_id = str(item.get("id", ""))
        task = item.get("question", "")
        broken_md = item.get("broken_md", "")
        rules_ref = item.get("rules_ref", [])

        # Construct prompt
        if skill_content.strip():
            skill_section = f"## Skill\n{skill_content.strip()}\n\n"
        else:
            skill_section = ""

        system_prompt = (
            "You are an expert AI agent that corrects formatting issues in Markdown documents to comply with strict repository rules.\n"
            f"{skill_section}"
            "You must output only the corrected Markdown content. Do not include any explanations, introduction, markdown block wrappers (like ```markdown), or additional text."
        )

        user_prompt = (
            f"## Task\n{task}\n\n"
            f"## Broken Document\n{broken_md}\n\n"
            f"## Rules Reference\n{', '.join(rules_ref) if rules_ref else 'Follow standard Seneca rules.'}\n\n"
            "Please output the corrected document."
        )

        # Call target backend
        pred_text = ""
        fail_reason = ""
        try:
            pred_text, _ = chat_target(
                system=system_prompt,
                user=user_prompt,
                max_completion_tokens=self.max_completion_tokens,
                retries=3,
                stage="rollout",
            )
        except Exception as e:
            fail_reason = f"LLM error: {e}"

        # Clean wrappers if target model generated them anyway
        pred_text = pred_text.strip()
        if pred_text.startswith("```"):
            newline_idx = pred_text.find("\n")
            if newline_idx != -1:
                pred_text = pred_text[newline_idx:].strip()
            else:
                pred_text = pred_text[3:].strip()
        if pred_text.endswith("```"):
            pred_text = pred_text[:-3].strip()

        # Grade using validator
        hard = 0
        soft = 0.0
        if not fail_reason:
            if grader_lib is None:
                fail_reason = "skillopt_grader_single import failed"
            else:
                with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as f:
                    f.write(pred_text)
                    temp_path = f.name
                try:
                    # Rel path corresponds to type of file (e.g. project-log, skill etc.)
                    rel_path = "08_Templates/temp.md"
                    if "changelog" in task.lower():
                        rel_path = "01_Engine_SOP/CHANGELOG.md"
                    elif "operating" in task.lower() or "rules" in task.lower():
                        rel_path = "00_Context_Engineering/agent-operating-rules.md"

                    errors = grader_lib.validate_single_file(temp_path, rel_path)
                    if errors:
                        fail_reason = "; ".join(errors)
                    else:
                        hard = 1
                        soft = 1.0
                finally:
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # Save predictions for reflect/analysis
        pred_item_dir = os.path.join(out_dir, "predictions", item_id)
        os.makedirs(pred_item_dir, exist_ok=True)
        with open(os.path.join(pred_item_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(system_prompt)
        with open(os.path.join(pred_item_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(user_prompt)
        with open(os.path.join(pred_item_dir, "predicted_answer.txt"), "w", encoding="utf-8") as f:
            f.write(pred_text)

        conversation = [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": pred_text}
        ]
        if fail_reason:
            eval_detail = f"[EVALUATION RESULT]\nValidation failed with errors:\n{fail_reason}"
        else:
            eval_detail = "[EVALUATION RESULT]\nValidation passed successfully."
        conversation.append({"role": "system", "content": eval_detail})

        with open(os.path.join(pred_item_dir, "conversation.json"), "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

        return {
            "id": item_id,
            "hard": hard,
            "soft": soft,
            "predicted_answer": pred_text,
            "question": task,
            "task_description": task,
            "task_type": item.get("task_type", "markdown-compliance"),
            "fail_reason": fail_reason,
        }

    def rollout(
        self,
        env_manager,
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict]:
        items: list[dict] = env_manager
        results: list[dict] = []
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            futs = [executor.submit(self._rollout_one, item, skill_content, out_dir) for item in items]
            for fut in futs:
                results.append(fut.result())
        return results

    def reflect(
        self,
        results: list[dict],
        skill_content: str,
        out_dir: str,
        **kwargs,
    ) -> list[dict | None]:
        prediction_dir = kwargs.get("prediction_dir", os.path.join(out_dir, "predictions"))
        patches_dir = kwargs.get("patches_dir", os.path.join(out_dir, "patches"))
        return run_minibatch_reflect(
            results=results,
            skill_content=skill_content,
            prediction_dir=prediction_dir,
            patches_dir=patches_dir,
            workers=self.analyst_workers,
            failure_only=self.failure_only,
            minibatch_size=self.minibatch_size,
            edit_budget=self.edit_budget,
            random_seed=kwargs.get("random_seed"),
            error_system=self.get_error_minibatch_prompt(),
            success_system=self.get_success_minibatch_prompt(),
            step_buffer_context=kwargs.get("step_buffer_context", ""),
            update_mode=getattr(self, "_cfg", {}).get("skill_update_mode", "patch"),
        )

    def get_task_types(self) -> list[str]:
        seen: list[str] = []
        all_items = (
            self.dataloader.train_items
            + self.dataloader.val_items
            + self.dataloader.test_items
        )
        for item in all_items:
            tt = str(item.get("task_type") or "markdown-compliance")
            if tt not in seen:
                seen.append(tt)
        return seen or ["markdown-compliance"]
