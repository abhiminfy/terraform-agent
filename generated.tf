```terraform
resource "aws_instance" "example" {
  ami = "ami-09ac0b140f63d3458" # Ubuntu 22.04 LTS us-east-1
  instance_type = "t2.micro"
}

resource "aws_db_instance" "default" {
  allocated_storage    = 20
  engine               = "mysql"
  engine_version        = "8.0"
  instance_class       = "db.t3.micro"
  identifier = "terraform-mysql-db"
  name                 = "mydb"
  username = "adminuser"
  password = "MyS3cur3P@ssw0rd!"
  parameter_group_name = "default.mysql8.0"
  skip_final_snapshot  = true

}
```
