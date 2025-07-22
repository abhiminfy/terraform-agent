resource "aws_instance" "example" {
 ami           = data.aws_ami.ubuntu_latest.id
 instance_type = "t2.micro"
 subnet_id = data.aws_subnet.default.id # Replace with your preferred subnet
 key_name = "terraform-key" # Replace with your key pair name

  vpc_security_group_ids = ["sg-xxxxxx"]

 tags = {
    Name = "example-instance"
 }
}


data "aws_ami" "ubuntu_latest" {
 most_recent = true
 owners      = ["099720109477"] # Canonical

 filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
 }
 filter {
    name   = "virtualization-type"
    values = ["hvm"]
 }

}

data "aws_subnet" "default" {
 availability_zone = "us-east-1a" # Replace with your preferred AZ
 default_for_az = true
 state = "available"
}




resource "aws_security_group" "allow_ssh" {
 name        = "allow_ssh"
 description = "Allow SSH inbound traffic"

 ingress {
    description      = "SSH from anywhere"
    from_port        = 22
    to_port          = 22
    protocol        = "tcp"
    cidr_blocks      = ["0.0.0.0/0"] # Open to the world for testing purposes, restrict in production
 }

 egress {
 from_port        = 0
 to_port          = 0
 protocol        = "-1"
 cidr_blocks      = ["0.0.0.0/0"]
 }


 tags = {
    Name = "allow_ssh"
 }
}




output "public_ip" {
 value = aws_instance.example.public_ip
}