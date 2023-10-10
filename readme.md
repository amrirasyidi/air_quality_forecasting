# Data Source
1. [MCC-SINICA](https://github.com/MCC-SINICA/Using-Satellite-Data-on-Remote-Transportation)
2. [aqicn.org](https://aqicn.org/historical#!city:indonesia/jakarta/us-consulate/south)
3. [openaq](https://openaq.org/developers/platform-overview/)

# Notes

## Envs
Create yml file
`conda env export --no-builds | findstr -v "prefix" > environment.yml`

Install from yml file
`conda env create --name envname --file=environment.yml`
