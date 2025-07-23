resource "aws_instance" "example" {
  count                  = 3
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = "t2.medium"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = ["sg-xxxxxx"]

  root_block_device {
    volume_type = "gp3"
    volume_size = 50
    delete_on_termination = true
  }

  tags = {
    Name = "example-${count.index}"
  }
}

resource "aws_security_group" "allow_ssh" {
 name        = "allow_ssh"
 description = "Allow SSH inbound traffic"
 vpc_id = aws_default_vpc.default.id

 ingress {
   description      = "SSH from anywhere"
   from_port        = 22
   to_port          = 22
   protocol        = "tcp"
   cidr_blocks      = ["0.0.0.0/0"]
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


data "aws_ami" "amazon_linux" {
 most_recent = true
 owners      = ["amazon"]

 filter {
   name   = "name"
   values = ["amzn2-ami-hvm-*-x86_64-gp2"]
 }
 filter {
   name   = "virtualization-type"
   values = ["hvm"]
 }

}

resource "aws_key_pair" "deployer" {
  key_name = "terraform-key"
  public_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABAQD..." # Update with your public key file path.

}

data "aws_availability_zones" "available" {}

resource "aws_default_vpc" "default" {
 tags = {
   Name = "Default VPC"
 }
}