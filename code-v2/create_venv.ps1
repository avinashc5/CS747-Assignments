$venv_name="env747"

$python_executable = Read-Host "Enter the path to Python 3 executable: "

if ((Get-Command $python_executable -ErrorAction SilentlyContinue) -eq $null) 
{ 
  Write-Host "python_executable is not a valid executable. Please check the python_executable path used in the script."
  Write-Host "Virtual environment NOT created. Try again."
  exit
}

& $python_executable -m venv $venv_name

& .\env747\Scripts\Activate.ps1

& pip install -r requirements.txt

$out = "Virtual environment " + $venv_name + " created and libraries installed."
Write-Host $out




