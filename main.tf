resource "aws_instance" "example" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t2.micro"
  key_name = aws_key_pair.example.key_name


  vpc_security_group_ids = ["sg-xxxxxx"]

  tags = {
    Name = "ExampleAppServerInstance"
  }
}

resource "aws_security_group" "example" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"
  vpc_id      = aws_default_vpc.default.id

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

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}



resource "aws_key_pair" "example" {
  key_name = "terraform-key"
 public_key = file("~/.ssh/id_rsa.pub") # Update with your actual public key file
}


data "aws_default_vpc" "default" {
}


output "public_ip" {
  value = aws_instance.example.public_ip
}