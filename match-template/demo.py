# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os

def antinoise(image, path):
    colored = cv2.imread(path)
    template = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF)
    foo = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    foo[15:15+h, 15:15+w] = colored
    cv2.putText(foo, 'ANTI-NOISE', (15, 255), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255));
    (_, maxScore, _, maxPos) = cv2.minMaxLoc(res)
    cv2.rectangle(foo, maxPos, (maxPos[0]+w, maxPos[1]+h), (0, 255, 0), 2)
    cv2.putText(foo, '%.2fm'%(maxScore/1000000.0), maxPos, cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0));
    foo=cv2.resize(foo, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('antinoise-'+path, foo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def naive(image, path):
    colored = cv2.imread(path)
    template = cv2.cvtColor(colored, cv2.COLOR_BGR2GRAY)
    w, h = template.shape[::-1]
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    foo = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    foo[15:15+h, 15:15+w] = colored
    cv2.putText(foo, 'NAIVE', (15, 255), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 0));
    keypoints = np.where(res>=0.95)
    for pos in zip(*keypoints[::-1]):
        cv2.rectangle(foo, pos, (pos[0]+w, pos[1]+h), (0, 255, 0), 2)
    foo=cv2.resize(foo, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('naive-'+path, foo)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    for sample in [fname for fname in os.listdir('sample') if '.jpg' in fname]:
        image = cv2.imread(os.path.join('sample', sample))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for template in [fname for fname in os.listdir('suit') if '.jpg' in fname]:
            path = os.path.join('suit', template)
            naive(image, path)
        for template in [fname for fname in os.listdir('suit') if '.jpg' in fname]:
            path = os.path.join('suit', template)
            antinoise(image, path)

if __name__=='__main__':
    main()
