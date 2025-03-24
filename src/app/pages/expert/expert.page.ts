import { Component, OnInit } from '@angular/core';
import { RestAPIfromDjangoService, User } from '../../services/rest-apifrom-django.service';
@Component({
  selector: 'app-expert',
  templateUrl: './expert.page.html',
  styleUrls: ['./expert.page.scss'],
  standalone: false,
})
export class ExpertPage implements OnInit {
 currentUser: User | null = null

  constructor(private usersService: RestAPIfromDjangoService, ) { }

  ngOnInit() {
    this.currentUser = this.usersService.getCurrentUser()
  }



  
}
