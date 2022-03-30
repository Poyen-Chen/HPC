#!/usr/bin/env zsh

### Job name
#SBATCH --job-name=stream2

### File / path where STDOUT & STDERR will be written
### %J is the job id
#SBATCH --output=output_%J.txt

### Request CLAIX18 MPI partition
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

### Request node exclusively
#SBATCH --exclusive

### Specify your mail address
###SBATCH --mail-user=<specify_your_mail>

### Send a mail when job is done
###SBATCH --mail-type=END

taskset -c 10 ./stream2.exe | tee stream2.txt
