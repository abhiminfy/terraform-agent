resource "aws_instance" "example" {
 ami           = data.aws_ami.amazon_linux_2_latest.id
 instance_type = "t2.micro"
 subnet_id = data.aws_subnet.default.id # Assuming a default VPC and subnet exists. If not, you will need to create them.
  key_name = "terraform-key" # Replace with your actual key pair name

  vpc_security_group_ids = ["sg-xxxxxx"]

 tags = {
    Name = "example-instance"
 }
}



data "aws_ami" "amazon_linux_2_latest" {
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


data "aws_subnet" "default" {
 availability_zone = "us-east-1a" # Replace with desired availability zone
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