# Real Time Twitter Sentiment Analysis<br />
<p align="center">
  <img src="https://github.com/SidNimbalkar/CSYE7245FinalProject/blob/master/Images/logo.png">
</p>

### Collabarators 
Gurjot Kaur<br/>
Harshitha Sanikommu<br/>
Sid Nimbalkar

### Professor
Sri Krishnamurthy

### Project Proposal
[Google Document](add link)

[Codelab](https://codelabs-preview.appspot.com/?file_id=11guPZm2NIzOZI7QMATwpICSQLIaFqfXFUYzi_k8Gdj4#0)

### Project Presentation Links 
[Google Document](add link)

[Codelab](add link)

## Install instructions

### Create an Amazon Web Services (AWS) account


If you already have an account, skip this step.

Go to this [link](https://signin.aws.amazon.com/signin?redirect_uri=https%3A%2F%2Fportal.aws.amazon.com%2Fbilling%2Fsignup%2Fresume&client_id=signup) and follow the instructions.
You will need a valid debit or credit card. You will not be charged, it is only to validate your ID.


### Install AWS Command Line Interface (AWSCLI)

Install the AWS CLI Version 1 for your operating system. Please follow the appropriate link below based on your operating system.

* [macOS](https://docs.aws.amazon.com/cli/latest/userguide/install-macos.html)

* [Windows](https://docs.aws.amazon.com/cli/latest/userguide/install-windows.html#install-msi-on-windows)

** Please make sure you add the AWS CLI version 1 executable to your command line Path.
Verify that AWS CLI is installed correctly by running `aws --version`.

* You should see something similar to `aws-cli/1.17.0 Python/3.7.4 Darwin/18.7.0 botocore/1.14.0`.

#### Configuring the AWS CLI

You need to retrieve AWS credentials that allow your AWS CLI to access AWS resources.

1. Sign into the AWS console. This simply requires that you sign in with the email and password you used to create your account.
If you already have an AWS account, be sure to log in as the root user.
2. Choose your account name in the navigation bar at the top right, and then choose My Security Credentials.
3. Expand the Access keys (access key ID and secret access key) section.
4. Press Create New Access Key.
5. Press Download Key File to download a CSV file that contains your new AccessKeyId and SecretKey. Keep this file somewhere where you can find it easily.

Now, you can configure your AWS CLI with the credentials you just created and downloaded.

1. In your Terminal, run `aws configure`.

   i. Enter your AWS Access Key ID from the file you downloaded.\
   ii. Enter the AWS Secret Access Key from the file.\
   iii. For Default region name, enter `us-east-1`.\
   iv. For Default output format, enter `json`.

2. Run `aws s3 ls` in your Terminal. If your AWS CLI is configured correctly, you should see nothing (because you do not have any existing AWS S3 buckets) or if you have created AWS S3 buckets before, they will be listed in your Terminal window.

** If you get an error, then please try to configure your AWS CLI again.

### Get Twitter API Keys
1. Create a free Twitter user account, This will allow you to access the Twitter developer portal.

2. Navigate to [Twitter Dev Site](https://apps.twitter.com), sign in, and create a new application.
After that, fill out all the app details. 
Once you do this, you should have your access keys.


### Install Postman

Follow the instructions of your operating system:

[macOS](https://learning.postman.com/docs/postman/launching-postman/installation-and-updates/#installing-postman-on-mac)

[Windows](https://learning.postman.com/docs/postman/launching-postman/installation-and-updates/#installing-postman-on-windows)

### Install Docker

Install Docker Desktop. Use one of the links below to download the proper Docker application depending on your operating system. Create a DockerHub account if asked.

* For macOS, follow this [link](https://docs.docker.com/docker-for-mac/install/).

* For Windows 10 64-bit Home, follow this [link](https://docs.docker.com/docker-for-windows/install/)

 i.  Excecute the files "first.bat" and "second.bat" in order, as administrator.

 ii. Restart your computer.

 iii.Excecute the following commands in terminal, as administrator.
 
     ```
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /f /v EditionID /t REG_SZ /d "Professional"
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /f /v ProductName /t REG_SZ /d "Windows 10 Pro"
     ```
     
 iv. Follow this [link](https://docs.docker.com/docker-for-windows/install/) to install Docker.
 
 v.  Restart your computer, do not log out.

 vi. Excecute the following commands in terminal, as administrator.
 
     ```
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v EditionID /t REG_SZ /d "Core"\
     REG ADD "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion" /v ProductName /t REG_SZ /d "Windows 10 Home"
     ```

Open a Terminal window and type `docker run hello-world` to make sure Docker is installed properly . It should appear the following message:

`` Hello from Docker!``  
``This message shows that your installation appears to be working correctly.``

Finally, in the Terminal window excecute `docker pull tensorflow/tensorflow:2.1.0-py3-jupyter`.

### Install Anaconda

Follow the instructions for your operating system.

* For macOS, follow this [link](https://docs.anaconda.com/anaconda/install/mac-os/)
* For Windows, follow this [link](https://docs.anaconda.com/anaconda/install/windows/)


### Install Sublime

Follow the [instructions](https://www.sublimetext.com/3) for your operating system.\
If you already have a prefered text editor, skip this step.

### Install Kafka 

Follow the following [instructions](https://kafka.apache.org/quickstart) to install zookeeper and kafka on your system. <br />
Once done you can use the following commands to run the kafka server.

Start Zookeeper <br />
`
 bin/zookeeper-server-start.sh config/zookeeper.properties
`

Start Kafka <br />
`
bin/kafka-server-start.sh config/server.properties
`

### Install Druid (Windows not supported)

Follow the following [instructions](https://druid.apache.org/docs/latest/tutorials/index.html) to install Druid on your system. <br />

#### Pre-requisites <br />
- Java 8 (8u92+) or later
- Linux, Mac OS X, or other Unix-like OS (Windows is not supported)

### Install Turnilo (Windows not supported)

#### Pre-requisites <br />
- Node.js - 10.x or 8.x version.
- npm - 6.5.0 version.

Once you have the pre-requisite packages:

Install Turnilo distribution using npm. <br />
`
npm install -g turnilo
`

To connect to the existing Druid broker using --druid command line option. Turnilo will automatically introspect your Druid broker and figure out available datasets. <br />
`
turnilo --druid http[s]://druid-broker-hostname[:port]
`

## Run Sequence

1. Run requirements.txt
```
pip install -U -r requirements.txt
```
This command will instal all the required packages and update any older packages.

2. Now that we have our enviornment set up, we will create an S3 bucket.

Follow this [link](https://docs.aws.amazon.com/AmazonS3/latest/gsg/CreatingABucket.html) and create a S3 bucket. 

3. Scraping Tweets: To run the scraping pipeline follow the [instructions](https://github.com/SidNimbalkar/CSYE7245FinalProject/tree/master/ScrapingPipeline) in the Scraping Pipeline folder.





