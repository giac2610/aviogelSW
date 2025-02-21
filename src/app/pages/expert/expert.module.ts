import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { IonicModule } from '@ionic/angular';

import { ExpertPageRoutingModule } from './expert-routing.module';

import { ExpertPage } from './expert.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    ExpertPageRoutingModule
  ],
  declarations: [ExpertPage]
})
export class ExpertPageModule {}
