import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { IonicModule } from '@ionic/angular';

import { SetupPageRoutingModule } from './setup-routing.module';

import { SetupPage } from './setup.page';
import { RouterModule } from '@angular/router';
import { VirtualKeyboardComponent } from "src/app/components/virtual-keyboard/virtual-keyboard.component";

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    RouterModule,
    SetupPageRoutingModule,
    VirtualKeyboardComponent
],
  declarations: [SetupPage]
})
export class SetupPageModule {}
