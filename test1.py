
global_vars = None
local_vars=None

with open("main.py") as f:
	print('*** Running main ***')
	code = compile(f.read(), "main.py", 'exec')
	exec(code, global_vars, local_vars)