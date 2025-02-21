
import { Component } from '@angular/core';

@Component({
  selector: 'tutorial-page',
  templateUrl: './tutorial.page.html',
  styleUrls: ['./tutorial.page.scss'],
  standalone: false
})
export class TutorialPage {
  steps = ["Start", "Step 1", "Step 2", "Step 3", "Finish"];
  currentStep = 0;

  nextStep() {
    if (this.currentStep < this.steps.length - 1) {
      this.currentStep++;
    }
  }

  prevStep() {
    if (this.currentStep > 0) {
      this.currentStep--;
    }
  }

}