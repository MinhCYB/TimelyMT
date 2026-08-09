.PHONY: help inspect-data fetch-talks acquire-data parse-data align-data time-data build-data manifest-data prepare-data prepare-dataset validate-data validate-dataset dataset-summary run-fixed-n run-fixed-time run-local-agreement generate-labels train-policy compare-models run-ablation evaluate test demo-api demo-web clean

ifeq ($(OS),Windows_NT)
PYTHONPATH_ENV := set PYTHONPATH=src&&
else
PYTHONPATH_ENV := PYTHONPATH=src
endif

help:
	@echo TimelyMT workflow targets:
	@echo   inspect-data
	@echo   fetch-talks
	@echo   acquire-data ARGS="--talk ted-jeff-dean-ai-smart"
	@echo   parse-data ARGS="--provider ted --talk ted-jeff-dean-ai-smart"
	@echo   align-data ARGS="--talk ted-jeff-dean-ai-smart"
	@echo   time-data ARGS="--talk ted-jeff-dean-ai-smart"
	@echo   build-data ARGS="--talk ted-jeff-dean-ai-smart"
	@echo   manifest-data
	@echo   prepare-dataset ARGS="--resume"
	@echo   validate-dataset
	@echo   dataset-summary
	@echo   prepare-data
	@echo   validate-data
	@echo   run-fixed-n
	@echo   run-fixed-time
	@echo   run-local-agreement
	@echo   generate-labels
	@echo   train-policy
	@echo   compare-models
	@echo   run-ablation
	@echo   evaluate
	@echo   test
	@echo   demo-api
	@echo   demo-web
	@echo   clean

acquire-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.acquisition.cli $(ARGS)

parse-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.parsing.cli $(ARGS)

align-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.alignment.cli $(ARGS)

time-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.timing.cli $(ARGS)

build-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.canonical.cli $(ARGS)

manifest-data:
	$(PYTHONPATH_ENV) python -m timelymt.data.manifest.cli build

prepare-dataset:
	$(PYTHONPATH_ENV) python -m timelymt.data.pipeline.cli prepare $(ARGS)

validate-dataset:
	$(PYTHONPATH_ENV) python -m timelymt.data.pipeline.cli validate

dataset-summary:
	$(PYTHONPATH_ENV) python -m timelymt.data.pipeline.cli summary

test:
	$(PYTHONPATH_ENV) python -m unittest discover -s tests -p "test_*.py" -v

inspect-data fetch-talks prepare-data validate-data run-fixed-n run-fixed-time run-local-agreement generate-labels train-policy compare-models run-ablation evaluate demo-api demo-web clean:
	@echo "Not implemented yet."
