import getpass

from django.core.management.base import BaseCommand

from accounts.models import User


class Command(BaseCommand):
    help = 'Create an admin user interactively (handles passwords with special characters).'

    def handle(self, *args, **options):
        self.stdout.write('Create admin user\n')

        username = input('Username: ').strip()
        if not username:
            self.stderr.write('Username is required.')
            return

        if User.objects.filter(username=username).exists():
            self.stderr.write(f'User "{username}" already exists.')
            return

        email = input('Email (optional): ').strip()

        while True:
            password = getpass.getpass('Password: ')
            password2 = getpass.getpass('Password (again): ')
            if password != password2:
                self.stderr.write('Passwords do not match. Try again.\n')
                continue
            if len(password) < 6:
                self.stderr.write('Password must be at least 6 characters.\n')
                continue
            break

        User.objects.create_superuser(
            username=username,
            email=email,
            password=password,
            role='admin',
        )
        self.stdout.write(self.style.SUCCESS(f'Admin user "{username}" created successfully.'))
