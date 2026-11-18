PY ?= python
RUN = PYTHONPATH=src $(PY) -m superr run

.PHONY: all install canon states jev jev-plan balls stints fit rank validate report test clean clean-all

all:            ## full pipeline: stages 0-8
	$(RUN) all

install:
	$(PY) -m pip install -r requirements.txt

canon:    ; $(RUN) canon
states:   ; $(RUN) states
jev:      ; $(RUN) jev
balls:    ; $(RUN) balls
stints:   ; $(RUN) stints
fit:      ; $(RUN) fit
rank:     ; $(RUN) rank
validate: ; $(RUN) validate
report:   ; $(RUN) report

jev-plan:       ## estimate Jev requests and cost without calling the API
	PYTHONPATH=src $(PY) -m superr plan-jev

model:          ## re-run everything downstream of the Jev layer (after changing model settings)
	$(RUN) fit rank validate report

test:
	PYTHONPATH=src $(PY) -m pytest -q

clean:          ## remove derived tables and outputs (keeps the Jev cache)
	rm -rf data/interim/*.parquet data/interim/*.npz outputs/* report/figures/*.png

clean-all: clean
	rm -rf data/jev_cache/*
