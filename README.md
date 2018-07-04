Installation procedure forthcoming.

If you want to see all environments available in your anaconda prompt:
```
conda info --envs
```

## To Run

Open an Anaconda Prompt then run these commands:
```
cd Documents\GitHub\plantcv-sandbox
activate plantcv
```
Only need to activate the plantcv environment when you open a new anaconda prompt.

Runs the pipeline:
```
python pipeline.py --image="input-images\original.jpg" --outdir="output-images" --debug="print"
```
(omit the `--debug` argument to leave out the intermediate images)

## Handy Commands

Deletes all ".png" in this folder only (e.g. the intermediate images)
```
rm *.png
```

Deletes all ".png" in the `output-images` folder only
```
rm output-images\*.png
```

_Careful with `rm` -- no undo (no recycling bin)_
