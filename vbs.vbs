Dim durationHour
durationHour = InputBox("输入控制小时数（可以输入小数）","edited by uuukkk",5)
Dim durationLoops
durationLoops = CInt(CDbl(durationHour) * 12) + 1
Set wshShell = WScript.CreateObject("WScript.Shell")

for i = 0 to durationLoops
wshShell.SendKeys "{NUMLOCK}"
WScript.Sleep 30000
wshShell.SendKeys "{NUMLOCK}"
WScript.Sleep 180000
next

MsgBox "Script Over"