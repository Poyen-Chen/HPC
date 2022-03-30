#!/bin/zsh

### Job name
#SBATCH --job-name=hello-world

### File / path where STDOUT & STDERR will be written
### %J is the job id
#SBATCH --output=output_%J.txt

### Choose CLAIX18 MPI partition
#SBATCH --partition=c18m

### Request project lecture
### Runtime limit: 20 min
### Max 2 jobs from one user can run simultaneously
#SBATCH -A lect0067

### Request the time you need for execution in minutes
### Format hours:min:sec
#SBATCH --time=00:05:00

### Request vitual memory. M is the default and can therefore be omitted,
### but could also be K(ilo)|G(iga)|T(era)
#SBATCH --mem-per-cpu=512M

### Specify your mail address
###SBATCH --mail-user=<specify_your_mail>

### Send a mail when job is done
###SBATCH --mail-type=END

./helloWorld.out
