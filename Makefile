install:
	uv sync

run:
	uv run -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results

	uv run -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results

index:
	uv run -m src index




eval:
	uv run -m src evaluate \
	  --student_answer_path data/output/search_results/dataset_docs_public.json \
	  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json

	# Evaluate code
	uv run -m src evaluate \
	  --student_answer_path data/output/search_results/dataset_code_public.json \
	  --dataset_path data/datasets/AnsweredQuestions/dataset_code_public.json	


debug:
	uv run -m pdb -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
		--k 10 \
		--save_directory data/output/search_results
	uv run -m pdb -m src search_dataset \
		--dataset_path data/datasets/UnansweredQuestions/dataset_code_public.json \
		--k 10 \
		--save_directory data/output/search_results


clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete


fclean:
	rm -rf data/output/search_results
	rm -rf data/processed/*.pkl


lint:
	flake8 src
	mypy src \
		--warn-return-any \
		--warn-unused-ignores \
		--ignore-missing-imports \
		--disallow-untyped-defs \
		--check-untyped-defs
