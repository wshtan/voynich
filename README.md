# Welcome to some project

<b>
NOTE: Anyone is welcome. But do not plagarize.
</b>

This repository is forked from: https://github.com/alexbstl/MCMC-Deciphering

(Some files are deleted, and this repo has rebased and archived)

## Quick Start

To run the program, type:

```bash
# To test if it is prepared to run:
python3 ./app.backup.py -h

# Now run it to decipher some scrambled text:
python3 ./app.backup.py -i ./data/warpeace_input.txt -d ./data/shakespeare_scrambled.txt
```

where the file `warpeace_input.txt` serves as training data (a.k.a. _corpus_),
which stores a large amount of plain English. The file
`shakespeare_scrambled.txt` are the scrambled text which we wish to decipher.

If error reported, see "troubleshooting" part.


## Troubleshooting

First we check if the program can be run in our computer. Running the program requires one to have Python3 intepreter and the numpy package. To check whether Python3 is available, type:

```bash
python3 --version  # Or:
python  --version  # Or:
py      --version

# Should print something like `Python 3.8.10`
#                              ~~~~~~~^~~~~~
#                                     As long as this is `3` then
#                                     we're good to move on.
```

If all three commands has error reported, then maybe [install Python](https://wiki.python.org/moin/BeginnersGuide/Download). But if it is certain that Python3 has already installed (e.g., it runs fine on VSCode), then ask for help in the channel.

If Python version is 2, then [update Python](https://google.com/) or consider using virtual environment helpers such as [anaconda](https://www.anaconda.com/) or [poetry](https://python-poetry.org/).

Suppose Python3 is available and it can be run by typing `python3`. Now we check whether [`numpy`](https://numpy.org/) is available. To do so, type:

```bash
python3 -c "import numpy; print(numpy.__version__)"  # 1.22.3
```

If error reported, then [install numpy](https://numpy.org/install/).

Suppose numpy is now installed. If `python3 ./app.py -h` still has error reported, then ask for help in the channel.
