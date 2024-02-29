# Create directories
mkdir -p ./dataset/ISRUC_S3/ExtractedChannels
mkdir -p ./dataset/ISRUC_S3/RawData
echo 'Make dataset dir: ./dataset/ISRUC_S3'

# Download raw data
cd ./dataset/ISRUC_S3/RawData
counter=1
while [ $counter -le 10 ]
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$counter.rar
    unrar x $counter.rar
    ((counter++))
done
echo 'Download Data to "./dataset/ISRUC_S3/RawData" complete.'

# Download extracted channels data
cd ../ExtractedChannels
counter=1
while [ $counter -le 10 ]
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$counter.mat
    ((counter++))
done
echo 'Download ExtractedChannels to "./dataset/ISRUC_S3/ExtractedChannels" complete.'

