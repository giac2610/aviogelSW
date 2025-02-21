import { ComponentFixture, TestBed } from '@angular/core/testing';
import { ExpertPage } from './expert.page';

describe('ExpertPage', () => {
  let component: ExpertPage;
  let fixture: ComponentFixture<ExpertPage>;

  beforeEach(() => {
    fixture = TestBed.createComponent(ExpertPage);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
