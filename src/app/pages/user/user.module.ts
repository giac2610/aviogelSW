import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { IonicModule } from '@ionic/angular';

import { UserPage } from './user.page';
import { UserPageRoutingModule } from './user-routing.module';
import { RouterModule } from '@angular/router';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    RouterModule,
    UserPageRoutingModule
  ],
  declarations: [UserPage]
})
export class UserPageModule {}
