import subprocess

ps = subprocess.Popen('ps -ef', shell=True, stdout=subprocess.PIPE)
print(ps.stdout.readlines())