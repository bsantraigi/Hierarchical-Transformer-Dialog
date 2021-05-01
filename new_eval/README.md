# Steps for New Eval:

Source: https://github.com/budzianowski/multiwoz

Create a python2 environment first, and install the requirements.

1. simplejson

1. Unzip Multiwoz 2.0 to data/multi-woz
2. Run `python create_delex_data.py`
3. Then modify and run new_eval.ipynb as required.

```bash
cd data/
unzip MultiWOZ_2.0.zip
rm __MACOSX -rf
mv 'MULTIWOZ2 2' multi-woz
cd ../
python create_delex_data.py
```