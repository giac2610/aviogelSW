import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

import { IonicModule } from '@ionic/angular';

import { BlobSimulationPageRoutingModule } from './blob-simulation-routing.module';

import { BlobSimulationPage } from './blob-simulation.page';

@NgModule({
  imports: [
    CommonModule,
    FormsModule,
    IonicModule,
    BlobSimulationPageRoutingModule,
  ],
  declarations: [BlobSimulationPage]
})
export class BlobSimulationPageModule {}
