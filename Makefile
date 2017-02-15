#Makefile

.DEFAULT_GOAL := help

.PHONY: help
help:
	@echo Options:
	@echo
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'


.PHONY: run
run:
	time python3 ./app.py -i ./data/warpeace_input.txt -d ./data/shakespeare_scrambled.txt --iters=20000


.PHONY: test.transpositions
test.transpositions:
	time python3 ./app-1tp-vs-2tp.py -i ./data/warpeace_input.txt -d ./data/shakespeare_scrambled.txt --iters=20000
