resource "aws_instance" "example" {
 ami           = data.aws_ami.ubuntu.id
 instance_type = "t2.micro"
 subnet_id = data.aws_subnet.default.id 
  key_name = "terraform-key" # Replace with your key pair name


  vpc_security_group_ids = ["sg-087ec30e65e4381af"]

 tags = {
    Name = "example-instance"
  }
}



data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }

}



data "aws_subnet" "default" {

  availability_zone = "us-east-1a"
 default_for_az = true

}





resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
 description = "Allow SSH inbound traffic"


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
    ipv6_cidr_blocks = ["::/0"]
  }

  tags = {
    Name = "allow_ssh"
  }
}